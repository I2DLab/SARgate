[![RDKit](https://img.shields.io/badge/RDKit-powered-blue)](https://www.rdkit.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

# SARgate

SARgate is a molecular toolkit for chemical space and structure-activity relationship analysis.

## Table of Contents

- [Download SARgate](#download-sargate)
- [Run SARgate from Source Code](#run-sargate-from-source-code)
  - [Requirements](#requirements)
  - [Download the Source Code](#download-the-source-code)
  - [Recommended Installation: Conda / Miniconda](#recommended-installation-conda--miniconda)
  - [Alternative Installation: Python venv](#alternative-installation-python-venv)
  - [Launch SARgate](#launch-sargate)
  - [Start SARgate After the First Installation](#start-sargate-after-the-first-installation)
- [Project Resources](#project-resources)
- [Common Problems and Solutions](#common-problems-and-solutions)
- [Updating the Environment After Source Code Changes](#updating-the-environment-after-source-code-changes)
- [Recommended Minimal Workflow for Non-Programmers](#recommended-minimal-workflow-for-non-programmers)
- [Build the Executable](#build-the-executable)

## Download SARgate

If you only want to use SARgate, download the pre-built executable for your operating system from the Releases page:

[https://github.com/I2DLab/SARgate/releases](https://github.com/I2DLab/SARgate/releases)

### Available Builds

| Operating system | Build archive |
|---|---|
| Windows | `SARgate-win64.zip` |
| macOS Apple Silicon | `SARgate-mac-arm64.zip` |
| Linux x86_64 | `SARgate-linux-x86_64.zip` |

Installation and launch instructions for executable builds are provided in:

```text
README_EXECUTABLES.txt
```

## Run SARgate from Source Code

The instructions below are intended for users who want to run or modify SARgate from source code.

SARgate is a Python application. To run it from source, you need the items listed below.

### Requirements

- A local copy of the SARgate project folder.
- Python installed on your computer.
- A terminal.
- An isolated Python environment, created either with Python `venv` or with Conda / Miniconda.
- The project dependencies installed inside that environment.

> **Important note**
>
> SARgate depends on RDKit. RDKit can sometimes be difficult to install in a plain pip-only environment on some systems.
> For this reason, the recommended installation method is a Conda environment.
> A Python `venv` is still possible, but Conda is generally more reliable.

## Download the Source Code

First, place the SARgate source code on your computer.

If you received it as a ZIP archive, extract it to a folder of your choice. If you cloned it from a repository, make sure the full project folder is available locally.

In all the examples below, the SARgate folder is the folder that contains files and folders such as:

```text
launcher.py
assets/
data/
installer/
app/
```

The main application entry point is located at:

```text
app/main.py
```

The dependency file and packaging files are stored inside the `installer/` folder:

```text
installer/requirements.txt
installer/installer.py
installer/SARgate.spec
```

Open a terminal and move into the SARgate project folder before doing anything else.

### macOS or Linux

```bash
cd /full/path/to/SARgate
```

### Windows Command Prompt

```bat
cd C:\full\path\to\SARgate
```

### Windows PowerShell

```powershell
cd "C:\full\path\to\SARgate"
```

## Recommended Installation: Conda / Miniconda

If you do not already have Conda, install Miniconda first.

### 1. Install Miniconda

Go to the official Miniconda website and download the installer for your operating system:

[https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

Install it with the default options unless your institution has specific policies.

After installation, close and reopen the terminal so that the `conda` command becomes available.

To verify that Conda is correctly installed, run:

```bash
conda --version
```

If the command prints a version number, Conda is ready.

### 2. Create a Dedicated Environment

Create a new environment for SARgate.

Python 3.13 is the safest target for this project.

```bash
conda create -n sargate python=3.13
```

When Conda asks for confirmation, type:

```text
y
```

### 3. Activate the Environment

Activate the new environment:

```bash
conda activate sargate
```

Your terminal prompt should now show the environment name, usually as:

```text
(sargate)
```

### 4. Install the Project Dependencies

From inside the SARgate project folder, install the dependencies listed in `installer/requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r installer/requirements.txt
```

If installation completes without errors, SARgate is ready to launch.

### 5. If RDKit Fails to Install with pip

If the previous command fails because RDKit cannot be installed, use Conda packages instead.

Run:

```bash
conda install -c conda-forge rdkit
python -m pip install dearpygui networkx numpy openpyxl pandas Pillow plotly pynput reportlab requests scikit-learn scipy screeninfo
```

This mixed approach is often the most robust solution on many systems.

## Alternative Installation: Python venv

Use this method only if you prefer a standard Python virtual environment and if your platform can install all required packages successfully.

### 1. Install Python

Download and install Python from:

[https://www.python.org/downloads/](https://www.python.org/downloads/)

Recommended version: Python 3.13.

On Windows, during installation, make sure to enable the option:

```text
Add Python to PATH
```

After installation, open a new terminal and verify that Python is available.

#### macOS or Linux

```bash
python3 --version
```

#### Windows

```bat
python --version
```

### 2. Create the Virtual Environment

Move to the SARgate project folder, then create a virtual environment.

#### macOS or Linux

```bash
python3 -m venv .venv
```

#### Windows

```bat
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### macOS or Linux

```bash
source .venv/bin/activate
```

#### Windows Command Prompt

```bat
.venv\Scripts\activate.bat
```

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you may need to allow local scripts for the current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then close and reopen PowerShell, return to the SARgate folder, and activate the environment again.

### 4. Install the Dependencies

After activation, install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r installer/requirements.txt
```

If RDKit fails to install in a `venv`, switch to the Conda method described above.

## Launch SARgate

Once the environment is active and the dependencies are installed, start SARgate from the project root folder.

Recommended command:

```bash
python launcher.py
```

The launcher shows the splash screen, preloads the major modules, and then starts the main GUI.

If your system uses `python3` instead of `python`, use:

```bash
python3 launcher.py
```

You can also run the main application directly, without the splash screen:

```bash
python app/main.py
```

or:

```bash
python3 app/main.py
```

However, `launcher.py` remains the recommended entry point for normal source-based execution.

## Start SARgate After the First Installation

After the first installation, you do not need to reinstall the dependencies every time.

You only need to:

1. Open a terminal.
2. Move into the SARgate project folder.
3. Activate the same environment.
4. Run `launcher.py`.

### Conda

```bash
cd /full/path/to/SARgate
conda activate sargate
python launcher.py
```

### venv on macOS or Linux

```bash
cd /full/path/to/SARgate
source .venv/bin/activate
python3 launcher.py
```

### venv on Windows

```bat
cd C:\full\path\to\SARgate
.venv\Scripts\activate.bat
python launcher.py
```

## Project Resources

The application expects its internal resources to remain in the project structure.

In particular:

| Resource type | Location |
|---|---|
| Configuration files | `assets/config/` |
| Interface assets | `assets/` |
| Input and output working folders | `data/` |
| Source code modules | `app/` |

Do not move these folders unless you also update the code accordingly.

## Common Problems and Solutions

### `python`, `python3`, or `conda` is not recognized

The corresponding software is either not installed or not available in `PATH`.

Reinstall it and reopen the terminal.

### Environment activation fails

Make sure you are inside the SARgate project folder and that the environment was actually created.

### Dependency installation fails

First upgrade pip:

```bash
python -m pip install --upgrade pip
```

### RDKit fails to install

Use the Conda method.

### The GUI does not start

If the installation completed successfully but the GUI does not start, try launching:

```bash
python app/main.py
```

This can help determine whether the issue is in the splash launcher or in the main application startup.

### macOS asks for permissions

If you are on macOS and the system asks for permissions related to files, screen access, or application control, grant them if needed for normal execution.

## Updating the Environment After Source Code Changes

If the SARgate source code changes and the new version introduces additional dependencies, reactivate the environment and run again:

```bash
python -m pip install -r installer/requirements.txt
```

This ensures that the environment remains synchronized with the current project version.

## Recommended Minimal Workflow for Non-Programmers

If you are unfamiliar with programming and simply want the safest route to run the source code, follow this exact sequence:

1. Install Miniconda.
2. Open a terminal.
3. Move into the SARgate folder.
4. Run:

```bash
conda create -n sargate python=3.13
conda activate sargate
python -m pip install --upgrade pip
python -m pip install -r installer/requirements.txt
python launcher.py
```

## Build the Executable

If you want to create a packaged executable from the source code, SARgate includes a dedicated `installer/` folder with:

```text
installer/installer.py
installer/SARgate.spec
installer/requirements.txt
```

Before creating the executable, make sure that:

1. You are inside the SARgate project root folder.
2. Your Python or Conda environment is activated.
3. PyInstaller is installed in that environment.

To install PyInstaller, run:

```bash
python -m pip install pyinstaller
```

### Build Using `installer.py`

This is the simplest method.

From the SARgate project root folder, run:

```bash
python installer/installer.py
```

or, if your system uses `python3`:

```bash
python3 installer/installer.py
```

This script automatically switches to the project root, collects the `assets/` and `data/` folders, and runs PyInstaller with the correct arguments.

### Build Using the `.spec` File

If you prefer to build directly from the PyInstaller specification file, run:

```bash
pyinstaller installer/SARgate.spec
```

or:

```bash
python -m PyInstaller installer/SARgate.spec
```

This method uses the explicit build definition contained in `installer/SARgate.spec`.

### Where the Build Output Will Appear

After a successful build, PyInstaller usually creates:

| Folder | Description |
|---|---|
| `build/` | Temporary build files |
| `dist/` | Generated SARgate application bundle or executable |

The exact format depends on the operating system:

| Operating system | Expected output |
|---|---|
| macOS | A `SARgate.app` bundle and/or a folder inside `dist/` |
| Windows | A SARgate executable inside `dist/` |
| Linux | An executable inside `dist/` |

### If the Build Fails

If packaging fails, check the following points:

- PyInstaller is installed in the currently active environment.
- All SARgate runtime dependencies are installed first from `installer/requirements.txt`.
- You are launching the build command from the SARgate project root folder.
- The `assets/`, `data/`, and `app/` folders are present and not renamed.

If needed, reinstall the dependencies and then retry:

```bash
python -m pip install -r installer/requirements.txt
python -m pip install pyinstaller
python installer/installer.py
```
