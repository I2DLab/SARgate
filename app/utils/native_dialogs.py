"""
=================
native_dialogs.py
=================

Native file and folder dialog helpers for SARgate.

This module provides synchronous wrappers around operating-system file dialogs.
Dialogs run in a tiny isolated helper process so SARgate never mixes Dear PyGui
and tkinter in the same interpreter session. This keeps the file pickers stable
on macOS while preserving file-type filters and the standard system dialog UI.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Build command helpers
# 3. Run macOS native dialogs through osascript
# 4. Run isolated tkinter dialogs in a helper process
# 5. Expose public dialog helpers
# 6. Execute the tkinter helper entry point

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import json
import os
import subprocess
import sys
import threading
import time
from typing import Any


def _bring_process_to_front(pid: int) -> None:
    """
    Ask macOS to bring the dialog helper process to the front.

    Args:
        pid (int): Process identifier of the helper process.

    Returns:
        None: This helper silently ignores platform-specific failures.
    """
    if sys.platform != "darwin":
        return

    script = [
        'tell application "System Events"',
        f"set frontmost of first application process whose unix id is {pid} to true",
        "end tell",
    ]
    command = ["osascript"]
    for line in script:
        command.extend(["-e", line])

    try:
        subprocess.run(command, capture_output=True, text=True, check=False)
    except Exception:
        pass


def _normalise_start_dir(default_path: str | None) -> str:
    """
    Return a usable absolute start directory for native dialogs.

    Args:
        default_path (str | None): Preferred start path or directory.

    Returns:
        str: Existing absolute directory path suitable for a dialog.
    """
    path = os.path.expanduser(default_path or "")
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if path and os.path.isdir(path):
        return os.path.abspath(path)
    return os.path.expanduser("~")


def _build_filetypes(file_types: list[tuple[str, str]] | None) -> list[tuple[str, str]]:
    """
    Normalize file-type definitions for native open/save dialogs.

    Args:
        file_types (list[tuple[str, str]] | None): Optional list of
            `(label, pattern)` pairs such as `("SDF files", "*.sdf")`.

    Returns:
        list[tuple[str, str]]: Cleaned list of file-type pairs.
    """
    if not file_types:
        return [("All files", "*.*")]
    return [(str(label), str(pattern)) for label, pattern in file_types]


# -----------------------------------------------------------------------------
# 3. Run isolated tkinter dialogs in a helper process
# -----------------------------------------------------------------------------
def _run_tk_helper(kind: str, title: str, default_path: str, file_types: list[tuple[str, str]] | None = None, default_name: str = "") -> str | None:
    """
    Run a tkinter dialog in a separate Python process.

    Args:
        kind (str): Dialog kind: `"open_file"`, `"open_directory"`, or
            `"save_file"`.
        title (str): Dialog title.
        default_path (str): Starting directory for the dialog.
        file_types (list[tuple[str, str]] | None, optional): Allowed file-type
            patterns for open/save dialogs.
        default_name (str, optional): Suggested file name for save dialogs.

    Returns:
        str | None: Selected path, or `None` when canceled.
    """
    payload = {
        "kind": kind,
        "title": title,
        "default_path": default_path,
        "file_types": _build_filetypes(file_types),
        "default_name": default_name,
        "parent_pid": os.getpid(),
    }
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--tk-helper", json.dumps(payload)]
    else:
        command = [sys.executable, __file__, "--tk-helper", json.dumps(payload)]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return None

    value = (result.stdout or "").strip()
    return value or None


# -----------------------------------------------------------------------------
# 4. Expose public dialog helpers
# -----------------------------------------------------------------------------
def open_file_dialog(title: str, default_path: str = "", file_types: list[tuple[str, str]] | None = None) -> str | None:
    """
    Open a native file-selection dialog.

    Args:
        title (str): Dialog title or prompt text.
        default_path (str, optional): Preferred starting directory.
        file_types (list[tuple[str, str]] | None, optional): Allowed file-type
            patterns for filtering.

    Returns:
        str | None: Selected file path, or `None` when canceled.
    """
    start_dir = _normalise_start_dir(default_path)
    return _run_tk_helper("open_file", title, start_dir, file_types=file_types)


def open_directory_dialog(title: str, default_path: str = "") -> str | None:
    """
    Open a native directory-selection dialog.

    Args:
        title (str): Dialog title or prompt text.
        default_path (str, optional): Preferred starting directory.

    Returns:
        str | None: Selected directory path, or `None` when canceled.
    """
    start_dir = _normalise_start_dir(default_path)
    return _run_tk_helper("open_directory", title, start_dir)


def save_file_dialog(
    title: str,
    default_path: str = "",
    default_name: str = "",
    file_types: list[tuple[str, str]] | None = None,
) -> str | None:
    """
    Open a native save-file dialog.

    Args:
        title (str): Dialog title or prompt text.
        default_path (str, optional): Preferred starting directory.
        default_name (str, optional): Suggested output file name.
        file_types (list[tuple[str, str]] | None, optional): Allowed file-type
            patterns for filtering.

    Returns:
        str | None: Selected output path, or `None` when canceled.
    """
    start_dir = _normalise_start_dir(default_path)
    return _run_tk_helper("save_file", title, start_dir, file_types=file_types, default_name=default_name)


# -----------------------------------------------------------------------------
# 5. Execute the tkinter helper entry point
# -----------------------------------------------------------------------------
def _run_tk_helper_main(payload: dict[str, Any]) -> int:
    """
    Execute the isolated tkinter helper process for native dialogs.

    Args:
        payload (dict[str, Any]): Serialized dialog configuration received from
            the parent process.

    Returns:
        int: Process exit code.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    root.update()

    parent_pid = int(payload.get("parent_pid", 0) or 0)

    def _parent_is_alive(pid: int) -> bool:
        """
        Check whether the parent SARgate process is still alive.

        Args:
            pid (int): Parent process identifier.

        Returns:
            bool: `True` when the process still appears alive.
        """
        if pid <= 0:
            return True

        if sys.platform == "win32":
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32
                process = kernel32.OpenProcess(0x100000, 0, pid)
                if not process:
                    return False
                wait_code = kernel32.WaitForSingleObject(process, 0)
                kernel32.CloseHandle(process)
                return wait_code == 0x102
            except Exception:
                return True

        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _watch_parent() -> None:
        """
        Terminate the helper process if the parent SARgate process exits.

        Args:
            None.

        Returns:
            None: This watcher force-exits the helper when needed.
        """
        while True:
            if not _parent_is_alive(parent_pid):
                os._exit(0)
            time.sleep(0.2)

    threading.Thread(target=_watch_parent, daemon=True).start()

    kind = str(payload.get("kind", ""))
    title = str(payload.get("title", "Select"))
    start_dir = str(payload.get("default_path", ""))
    file_types = payload.get("file_types") or [("All files", "*.*")]
    default_name = str(payload.get("default_name", ""))

    _bring_process_to_front(os.getpid())

    selected = ""
    if kind == "open_file":
        selected = filedialog.askopenfilename(
            title=title,
            initialdir=start_dir,
            filetypes=file_types,
        )
    elif kind == "open_directory":
        selected = filedialog.askdirectory(title=title, initialdir=start_dir)
    elif kind == "save_file":
        selected = filedialog.asksaveasfilename(
            title=title,
            initialdir=start_dir,
            initialfile=default_name,
            filetypes=file_types,
        )

    root.destroy()
    if selected:
        print(selected)
    return 0


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "--tk-helper":
    sys.exit(_run_tk_helper_main(json.loads(sys.argv[2])))
