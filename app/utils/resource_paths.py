"""
========
resource_paths.py
========

Path and filesystem management utilities.

This module provides functions to resolve resource paths bundled with
the application executable, handling various packaging scenarios such as
PyInstaller's onefile/onedir modes and macOS .app structures. It also includes
a robust method to open local HTML files across different operating systems,
addressing common permission issues.
"""

# =============================================================================
# STEP MAP
# =============================================================================
# 1. Import module dependencies
# 2. Candidate bases
# 3. Resource path
# 4. Open html safely
# 5. Project root from here

# -----------------------------------------------------------------------------
# 1. Import module dependencies
# -----------------------------------------------------------------------------

import os
import sys
import shutil
import tempfile
import webbrowser
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# 2. Candidate bases
# -----------------------------------------------------------------------------
def _candidate_bases() -> Any:
    """
    Determine possible base directories where bundled resources may reside.
    
    Args:
        None.
    
    Returns:
        Any: Value returned by the routine.
    """

    bases = []

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bases.append(Path(meipass))

    try:
        here = Path(__file__).resolve()
        proj_root = _project_root_from_here(here)
        bases.append(proj_root)
    except Exception:
        pass

    if sys.platform == "darwin":
        try:
            exe = Path(sys.executable).resolve()
            for parent in exe.parents:
                if parent.suffix == ".app":
                    bases.append(parent / "Contents" / "Resources")
                    break
        except Exception:
            pass

    try:
        exe_dir = Path(sys.executable).resolve().parent
        bases.append(exe_dir)
        bases.append(exe_dir / "_internal")
    except Exception:
        pass

    try:
        bases.append(Path(__file__).resolve().parent)
    except Exception:
        bases.append(Path.cwd())

    # Deduplicate while preserving order
    seen, out = set(), []
    for b in bases:
        if b not in seen:
            seen.add(b)
            out.append(b)

    return out


# -----------------------------------------------------------------------------
# 3. Resource path
# -----------------------------------------------------------------------------
def resource_path(*relative_parts: str) -> Path:
    """
    Return the absolute Path to a bundled resource.

    Args:
        relative_parts (str): One or more path components.

    Returns:
        Path: Fully resolved path for use inside packaged or dev environments.
    """

    rel = Path(*relative_parts)

    if rel.is_absolute():
        return rel

    bases = _candidate_bases()
    for base in bases:
        cand = base / rel
        if cand.exists():
            return cand

    try:
        here = Path(__file__).resolve()
        proj_root = _project_root_from_here(here)
        return proj_root / rel
    except Exception:
        pass

    return (bases[0] if bases else Path.cwd()) / rel


# -----------------------------------------------------------------------------
# 4. User data path
# -----------------------------------------------------------------------------
def user_data_path(*relative_parts: str) -> Path:
    """
    Return a writable per-user SARgate data path.

    Packaged macOS apps must not write inside the .app bundle after signing.
    This helper centralizes the writable runtime location used for settings,
    logs, and small synchronization files.
    """
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "SARgate"
    elif os.name == "nt":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) / "SARgate" if appdata else Path.home() / "AppData" / "Roaming" / "SARgate"
    else:
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg_data_home) / "SARgate" if xdg_data_home else Path.home() / ".local" / "share" / "SARgate"

    target = base / Path(*relative_parts) if relative_parts else base
    directory = target if not target.suffix else target.parent
    directory.mkdir(parents=True, exist_ok=True)
    return target


# -----------------------------------------------------------------------------
# 5. Open html safely
# -----------------------------------------------------------------------------
def open_html_safely(html_path: Path) -> bool:
    """
    Open a local HTML file robustly across operating systems.

    Handles permission restrictions inside PyInstaller bundles on Windows by
    copying HTML files to a temporary directory before opening them.

    Args:
        html_path (Path): HTML file path.

    Returns:
        bool: True if the browser call succeeded, False otherwise.
    """

    try:
        p = html_path.resolve()
        if not p.exists():
            log_event("SYSTEM", f"[HTML] Not found: {p}", indent=1, level="ERROR")
            return False

        if os.name == "nt":
            base = Path(getattr(sys, "_MEIPASS", "")) if hasattr(sys, "_MEIPASS") else None
            if base and base in p.parents:
                tmp_dir = Path(tempfile.gettempdir(), "SARgate_html")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                dst = tmp_dir / p.name
                try:
                    shutil.copy2(p, dst)
                    p = dst.resolve()
                except Exception as e:
                    log_exception("SYSTEM", "[HTML] Copy to temp failed", e, indent=1)

        ok = webbrowser.open(p.as_uri())
        if ok:
            return True

        if os.name == "nt":
            try:
                os.startfile(str(p))
                return True
            except Exception as e:
                log_exception("SYSTEM", "[HTML] os.startfile failed", e, indent=1)

        return False

    except Exception as e:
        log_exception("SYSTEM", "[HTML] open_html_safely error", e, indent=1)
        return False


# -----------------------------------------------------------------------------
# 6. Project root from here
# -----------------------------------------------------------------------------
def _project_root_from_here(start: Path) -> Path:
    """
    Ascend the directory tree from `start` to find the SARgate project root.

    Root is defined as the first directory containing 'app/'.

    Args:
        start (Path): Starting path.

    Returns:
        Path: Project root directory.
    """

    cur = start.resolve()

    for _ in range(8):  # Safety limit
        if (cur / "app").is_dir():
            return cur
        nxt = cur.parent
        if nxt == cur:
            break
        cur = nxt

    return start.parent
from app.utils.app_logger import log_event, log_exception
