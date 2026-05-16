# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

PROJECT_ROOT = os.path.abspath(os.getcwd())
SPEC_DIR = os.path.join(PROJECT_ROOT, 'installer')
APP_ICON_ICO = os.path.join(PROJECT_ROOT, 'assets', 'icons', 'SARgate_icon.ico')
APP_ICON_PNG = os.path.join(PROJECT_ROOT, 'assets', 'icons', 'SARgate_icon.png')
PYINSTALLER_ICON = APP_ICON_ICO if sys.platform in {"win32", "darwin"} else APP_ICON_PNG
MACOS_TARGET_ARCHITECTURE = os.environ.get('SARGATE_MACOS_TARGET_ARCHITECTURE') if sys.platform == 'darwin' else None
MACOS_INFO_PLIST = {
    'CFBundleDocumentTypes': [
        {
            'CFBundleTypeName': 'SARgate input files',
            'CFBundleTypeRole': 'Editor',
            'LSHandlerRank': 'Alternate',
            'CFBundleTypeExtensions': ['sdf', 'csv', 'tsv', 'xlsx', 'smi', 'txt'],
            'LSItemContentTypes': [
                'it.unipg.i2dlab.sargate.sdf',
                'it.unipg.i2dlab.sargate.smi',
                'public.comma-separated-values-text',
                'public.tab-separated-values-text',
                'org.openxmlformats.spreadsheetml.sheet',
                'public.plain-text',
            ],
        },
    ],
    'UTImportedTypeDeclarations': [
        {
            'UTTypeIdentifier': 'it.unipg.i2dlab.sargate.sdf',
            'UTTypeDescription': 'Structure-data file',
            'UTTypeConformsTo': ['public.text'],
            'UTTypeTagSpecification': {'public.filename-extension': ['sdf']},
        },
        {
            'UTTypeIdentifier': 'it.unipg.i2dlab.sargate.smi',
            'UTTypeDescription': 'SMILES file',
            'UTTypeConformsTo': ['public.text'],
            'UTTypeTagSpecification': {'public.filename-extension': ['smi']},
        },
    ],
} if sys.platform == 'darwin' else {}


def _collect_windows_expat_binaries():
    """
    Collect pyexpat and its native DLL dependencies explicitly for Windows,
    especially when building from Conda environments.
    """
    binaries = []
    seen = set()

    if sys.platform != "win32":
        return binaries

    roots = []
    for candidate in [sys.prefix, sys.base_prefix, os.path.dirname(sys.executable)]:
        if candidate and os.path.isdir(candidate) and candidate not in roots:
            roots.append(candidate)

    search_specs = [
        ("DLLs", "pyexpat.pyd"),
        ("Library/bin", "libexpat.dll"),
        ("Library/bin", "expat.dll"),
        ("", "libexpat.dll"),
        ("", "expat.dll"),
    ]

    for root in roots:
        for subdir, filename in search_specs:
            source = os.path.join(root, subdir, filename) if subdir else os.path.join(root, filename)
            if os.path.isfile(source):
                key = os.path.normcase(os.path.abspath(source))
                if key not in seen:
                    binaries.append((source, "."))
                    seen.add(key)

    return binaries


def _collect_root_documents():
    """
    Include root-level user-facing documents in the packaged application.
    """
    docs = []
    for filename in ['BUILD.md', 'README.md', 'README_EXECUTABLES.md', 'LICENSE']:
        source = os.path.join(PROJECT_ROOT, filename)
        if os.path.isfile(source):
            docs.append((source, '.'))
    return docs


datas = [
    (os.path.join(PROJECT_ROOT, 'app', 'analysis', 'chemspace', 'html'), 'app/analysis/chemspace/html'),
    (os.path.join(PROJECT_ROOT, 'app', 'analysis', 'tools', 'molecule_sketcher.py'), 'app/analysis/tools'),
    (os.path.join(PROJECT_ROOT, 'assets'), 'assets'),
    (os.path.join(PROJECT_ROOT, 'data'), 'data'),
]
datas += _collect_root_documents()
binaries = _collect_windows_expat_binaries()
hiddenimports = [
    'app.main',
    'PIL.ImageTk',
    'PIL._tkinter_finder',
    'pyexpat',
    'xml.parsers.expat',
    'rdkit.Chem.Descriptors',
    'rdkit.Chem.rdFingerprintGenerator',
    'rdkit.Chem.MolStandardize',
]
datas += collect_data_files('rdkit')
datas += collect_data_files('plotly')
datas += collect_data_files('reportlab')
hiddenimports += collect_submodules('app')
hiddenimports += collect_submodules('PIL')
hiddenimports += collect_submodules('openpyxl')
hiddenimports += collect_submodules('rdkit')
hiddenimports += collect_submodules('sklearn')
hiddenimports += collect_submodules('scipy')


a = Analysis(
    [os.path.join(PROJECT_ROOT, 'launcher.py')],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SARgate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=(sys.platform == 'darwin'),
    target_arch=MACOS_TARGET_ARCHITECTURE,
    codesign_identity=None,
    entitlements_file=None,
    icon=[PYINSTALLER_ICON],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SARgate',
)
app = BUNDLE(
    coll,
    name='SARgate.app',
    icon=APP_ICON_ICO,
    bundle_identifier='it.unipg.i2dlab.sargate',
    info_plist=MACOS_INFO_PLIST,
)
