# Building SARgate
 
This document explains how to create a packaged SARgate executable from the source code.

Executable builds are created with PyInstaller using the files stored in the `installer/` folder.

---

## Requirements

Before creating the executable, make sure that:

1. you are inside the SARgate project root folder;
2. your Python, Conda, or Miniconda environment is activated;
3. all runtime dependencies are installed from `installer/requirements.txt`;
4. PyInstaller is installed in the active environment;
5. the `assets/`, `data/`, and `app/` folders are present and not renamed.

Install the runtime dependencies:

```bash
python -m pip install -r installer/requirements.txt
```

Install PyInstaller:

```bash
python -m pip install pyinstaller
```

---

## Build Using `installer.py`

This is the recommended build method.

From the SARgate project root folder, run:

```bash
python installer/installer.py
```

If your system uses `python3` instead of `python`, run:

```bash
python3 installer/installer.py
```

The script automatically switches to the project root, collects the required `assets/` and `data/` folders, and runs PyInstaller with the correct arguments.

---

## Build Using the `.spec` File

You can also build SARgate directly from the PyInstaller specification file.

Run:

```bash
pyinstaller installer/SARgate.spec
```

or:

```bash
python -m PyInstaller installer/SARgate.spec
```

This method uses the explicit build definition contained in `installer/SARgate.spec`.

---

## Build Output

After a successful build, PyInstaller usually creates:

```text
build/
dist/
```

The exact output format depends on the operating system:

| Operating System | Typical Output |
| --- | --- |
| macOS | `SARgate.app` bundle and/or a folder inside `dist/` |
| Windows | SARgate executable inside `dist/` |
| Linux | SARgate executable inside `dist/` |

---

## If the Build Fails

If packaging fails, check the following points:

- PyInstaller is installed in the currently active environment;
- all SARgate runtime dependencies are installed from `installer/requirements.txt`;
- the build command is being launched from the SARgate project root folder;
- the `assets/`, `data/`, and `app/` folders are present and not renamed.

If needed, reinstall the dependencies and retry the build:

```bash
python -m pip install -r installer/requirements.txt
python -m pip install pyinstaller
python installer/installer.py
```

---

## Related Files

The build system uses the following files:

```text
installer/installer.py
installer/SARgate.spec
installer/requirements.txt
```

For executable download and launch instructions, see:

[README_EXECUTABLES.md](README_EXECUTABLES.md)
