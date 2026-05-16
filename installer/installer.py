"""
============
installer.py
============

Build script for packaging SARgate using PyInstaller.

This file defines the build configuration that assembles all assets, fonts,
HTML templates, and input/output folders into a redistributable application.
It collects required data and submodules from libraries such as RDKit,
Plotly, SciPy, scikit-learn, and ReportLab, producing a portable
`SARgate` bundle (app folder or executable) for macOS, Windows, and Linux.
"""

import os
import platform
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
import PyInstaller.__main__


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)


# === BUILD OPTIONS ===
# macOS target architectures:
# - "arm64": Apple Silicon only (M1/M2/M3/M4)
# - "x86_64": Intel Mac only
# - "universal2": one app for Apple Silicon + Intel, if all native libraries support it
# - None: PyInstaller default for the current Python/environment
MACOS_TARGET_ARCHITECTURE = "arm64"


PROJECT_ROOT_PATH = Path(PROJECT_ROOT)
CLEANUP_SCAN_DIR_NAMES = ("app", "assets", "installer", "install")
COPY_NAME_RE = re.compile(r"(^|[\s_\-\(])(copy|copia|copie)(\)|$|[\s_\-\.])", re.IGNORECASE)
DEFAULT_SETTINGS = {
    "input_directory": "",
    "results_directory": "",
    "predictions_directory": "",
    "show_tab_icons": False,
    "tab_button_size": 40,
    "font": "DejaVu Sans",
    "font_scale": 1.0,
    "theme_name": "SARgate",
    "colormap_continuous": "Aurora",
    "colormap_discrete": "Starfield",
    "pca_point_size": "Small",
    "pca_mcs_timeout": "10s",
    "pca_mcs_features": True,
    "umap_point_size": "Small",
    "umap_mcs_timeout": "10s",
    "umap_mcs_features": True,
    "tsne_point_size": "Small",
    "tsne_mcs_timeout": "10s",
    "tsne_mcs_features": True,
    "slith_record": 4,
}


def _is_copy_name(path: Path) -> bool:
    """
    Return True for accidental duplicate names such as 'file copy.py'.
    """
    return bool(COPY_NAME_RE.search(path.name))


def _delete_cleanup_path(path: Path) -> None:
    """
    Delete a generated or accidental local file before packaging.
    """
    relative = path.relative_to(PROJECT_ROOT_PATH)
    print(f"Removing: {relative}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _iter_cleanup_candidates() -> list[Path]:
    """
    Return cleanup candidates in child-before-parent order.
    """
    candidates: set[Path] = set()

    for path in PROJECT_ROOT_PATH.iterdir():
        if path.name == "__pycache__" or path.name == ".DS_Store" or _is_copy_name(path):
            candidates.add(path)

    for dirname in CLEANUP_SCAN_DIR_NAMES:
        root = PROJECT_ROOT_PATH / dirname
        if not root.exists() or not root.is_dir():
            continue
        for current_root, dirnames, filenames in os.walk(root):
            current = Path(current_root)

            for dirname in dirnames:
                path = current / dirname
                if dirname == "__pycache__" or _is_copy_name(path):
                    candidates.add(path)

            for filename in filenames:
                path = current / filename
                if filename == ".DS_Store" or _is_copy_name(path):
                    candidates.add(path)

    return sorted(candidates, key=lambda item: len(item.parts), reverse=True)


def _write_default_settings() -> None:
    """
    Regenerate assets/config/settings.ssf with clean default values.
    """
    settings_path = PROJECT_ROOT_PATH / "assets" / "config" / "settings.ssf"
    print(f"Resetting: {settings_path.relative_to(PROJECT_ROOT_PATH)}")
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = settings_path.with_suffix(f"{settings_path.suffix}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_SETTINGS, f, indent=4, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(settings_path)


def _write_default_recent_files() -> None:
    """
    Regenerate assets/config/recent_files.srf with an empty recent-files list.
    """
    recent_files_path = PROJECT_ROOT_PATH / "assets" / "config" / "recent_files.srf"
    print(f"Resetting: {recent_files_path.relative_to(PROJECT_ROOT_PATH)}")
    recent_files_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = recent_files_path.with_suffix(f"{recent_files_path.suffix}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"paths": []}, f, indent=4, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(recent_files_path)


def _run_pre_build_cleanup() -> None:
    """
    Clean local generated files and reset user-dependent settings before build.
    """
    print("Running pre-build cleanup...")
    candidates = _iter_cleanup_candidates()
    if not candidates:
        print("Nothing to clean.")
    else:
        for path in candidates:
            if path.exists():
                _delete_cleanup_path(path)
    _write_default_settings()
    _write_default_recent_files()
    print("Cleanup complete.")


_run_pre_build_cleanup()


def _normalized_architecture_label() -> str:
    """
    Return a compact architecture label for the release archive name.
    """
    if sys.platform == "darwin" and MACOS_TARGET_ARCHITECTURE:
        return str(MACOS_TARGET_ARCHITECTURE)

    machine = platform.machine().strip().lower() or "unknown"
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }
    return aliases.get(machine, machine)


def _platform_label() -> str:
    """
    Return the platform label used in the release archive name.
    """
    if sys.platform == "darwin":
        return "mac"
    if os.name == "nt":
        return "windows"
    return "linux"


def _build_output_path(dist_dir: str) -> str:
    """
    Locate the platform-specific PyInstaller output to archive.
    """
    if sys.platform == "darwin":
        candidates = [
            os.path.join(dist_dir, "SARgate.app"),
            os.path.join(PROJECT_ROOT, "SARgate.app"),
        ]
    else:
        candidates = [
            os.path.join(dist_dir, "SARgate"),
            os.path.join(dist_dir, "SARgate.exe"),
        ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Could not find SARgate build output in {dist_dir}")


def _clean_zip_metadata(archive_path: str) -> None:
    """
    Remove macOS metadata entries from the release zip archive.
    """
    source_path = Path(archive_path)
    tmp_path = source_path.with_suffix(f"{source_path.suffix}.tmp")

    with zipfile.ZipFile(source_path, "r") as source_zip:
        entries = source_zip.infolist()
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as clean_zip:
            for info in entries:
                normalized_name = info.filename.replace("\\", "/")
                parts = normalized_name.split("/")
                if (
                    "__MACOSX" in parts
                    or any(part == ".DS_Store" for part in parts)
                    or any(part.startswith("._") for part in parts)
                ):
                    continue
                clean_zip.writestr(info, source_zip.read(info.filename))

    tmp_path.replace(source_path)


def _create_release_archive(dist_dir: str) -> str:
    """
    Create the redistributable zip archive for the current platform.
    """
    build_output = _build_output_path(dist_dir)
    archive_stem = f"SARgate-{_platform_label()}-{_normalized_architecture_label()}"
    archive_path = os.path.join(PROJECT_ROOT, f"{archive_stem}.zip")
 
    if os.path.exists(archive_path):
        os.remove(archive_path)

    if sys.platform == "darwin":
        subprocess.run(
            ["ditto", "-c", "-k", "--keepParent", build_output, archive_path],
            cwd=PROJECT_ROOT,
            check=True,
        )
    else:
        parent_dir = os.path.dirname(build_output)
        base_name = os.path.basename(build_output)
        shutil.make_archive(
            os.path.join(PROJECT_ROOT, archive_stem),
            "zip",
            root_dir=parent_dir,
            base_dir=base_name,
        )

    _clean_zip_metadata(archive_path)
    return archive_path


def _resolve_build_paths() -> tuple[str, str]:
    """
    Choose dist/work paths, falling back on Windows if the previous dist is locked.
    """
    default_dist = os.path.join(PROJECT_ROOT, "dist")
    default_work = os.path.join(PROJECT_ROOT, "build")

    if os.name != "nt":
        return default_dist, default_work

    target_dir = os.path.join(default_dist, "SARgate")
    if not os.path.exists(target_dir):
        return default_dist, default_work

    try:
        shutil.rmtree(target_dir)
        return default_dist, default_work
    except Exception:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_dist = os.path.join(PROJECT_ROOT, f"dist_{stamp}")
        fallback_work = os.path.join(PROJECT_ROOT, f"build_{stamp}")
        print(
            f"Warning: existing build is locked on Windows; "
            f"using fallback dist folder: {fallback_dist}"
        )
        return fallback_dist, fallback_work


sep = os.pathsep  # ':' on mac/Linux, ';' on Windows
dist_path, work_path = _resolve_build_paths()


pyinstaller_args = [
    os.path.join(PROJECT_ROOT, "installer", "SARgate.spec"),
    "--noconfirm",
    "--distpath", dist_path,
    "--workpath", work_path,
]

if sys.platform == "darwin" and MACOS_TARGET_ARCHITECTURE:
    os.environ["SARGATE_MACOS_TARGET_ARCHITECTURE"] = MACOS_TARGET_ARCHITECTURE
else:
    os.environ.pop("SARGATE_MACOS_TARGET_ARCHITECTURE", None)

PyInstaller.__main__.run(pyinstaller_args)

archive_path = _create_release_archive(dist_path)
print(f"Created release archive: {archive_path}")
