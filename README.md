# SARgate

[![RDKit](https://img.shields.io/badge/RDKit-powered-blue)](https://www.rdkit.org/)
[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Releases](https://img.shields.io/badge/Downloads-GitHub%20Releases-informational)](https://github.com/I2DLab/SARgate/releases)

![SARgate interface](assets/screenshots/sargate-main.png)

SARgate is a molecular toolkit for chemical space and structure-activity relationship analysis.

It can be used either as a pre-built executable or directly from source code.


---

## Table of Contents

- [SARgate](#sargate)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Quick Start](#quick-start)
  - [Download SARgate](#download-sargate)
  - [Supported Platforms](#supported-platforms)
  - [Source Code](#source-code)
  - [Download the Source Code](#download-the-source-code)
  - [Recommended Installation: Conda / Miniconda](#recommended-installation-conda--miniconda)
    - [Install Miniconda](#install-miniconda)
    - [Create a Dedicated Environment](#create-a-dedicated-environment)
    - [Activate the Environment](#activate-the-environment)
    - [Install the Project Dependencies](#install-the-project-dependencies)
    - [If RDKit Fails to Install with pip](#if-rdkit-fails-to-install-with-pip)
  - [Alternative Installation: Python venv](#alternative-installation-python-venv)
    - [Install Python](#install-python)
    - [Create the Virtual Environment](#create-the-virtual-environment)
    - [Activate the Virtual Environment](#activate-the-virtual-environment)
    - [Install the Dependencies](#install-the-dependencies)
  - [Launch SARgate](#launch-sargate)
  - [Starting SARgate Again](#starting-sargate-again)
  - [Project Resources](#project-resources)
  - [Repository Structure](#repository-structure)
  - [Common Problems and Solutions](#common-problems-and-solutions)
    - [`python`, `python3`, or `conda` is not recognized](#python-python3-or-conda-is-not-recognized)
    - [Environment activation fails](#environment-activation-fails)
    - [Dependency installation fails](#dependency-installation-fails)
    - [The GUI does not start](#the-gui-does-not-start)
    - [macOS asks for permissions](#macos-asks-for-permissions)
  - [Updating the Environment After Source Code Changes](#updating-the-environment-after-source-code-changes)
  - [Recommended Minimal Workflow for Non-Programmers](#recommended-minimal-workflow-for-non-programmers)
  - [Building the Executable](#building-the-executable)
    - [Build Using `installer.py`](#build-using-installerpy)
    - [Build Using the `.spec` File](#build-using-the-spec-file)
    - [Build Output](#build-output)
    - [If the Build Fails](#if-the-build-fails)
  - [Third-party Software](#third-party-software)
  - [Contributing](#contributing)
  - [Citation](#citation)
  - [License](#license)

---

## Features

SARgate provides a local desktop workflow for molecular analysis tasks related to chemical space and structure-activity relationships.

The project includes:

- source-based execution through `launcher.py`;
- a main application entry point in `app/main.py`;
- a structured application layout with dedicated `app/`, `assets/`, `data/`, and `installer/` folders;
- dependency management through `installer/requirements.txt`;
- executable packaging support through PyInstaller;
- platform-specific release archives for Windows, macOS Apple Silicon, and Linux.

---

## Quick Start

The fastest way to use SARgate is to download a pre-built executable from the GitHub Releases page:

<https://github.com/I2DLab/SARgate/releases>

After downloading the archive for your operating system, extract it and follow the executable-specific instructions provided in:

[README_EXECUTABLES.md](README_EXECUTABLES.md)

For source-based execution, Conda or Miniconda is recommended because SARgate depends on RDKit.

---

## Download SARgate

If you only want to use SARgate, download the pre-built executable for your operating system from the Releases page:

<https://github.com/I2DLab/SARgate/releases>

Available builds:

| Operating System | Build Archive |
| --- | --- |
| Windows 64-bit | `SARgate-win64.zip` |
| macOS Apple Silicon | `SARgate-mac-arm64.zip` |
| Linux x86_64 | `SARgate-linux-x86_64.zip` |

Installation and launch instructions for executable builds are provided in:

[README_EXECUTABLES.md](README_EXECUTABLES.md)

---

## Supported Platforms

SARgate provides executable builds for:

- Windows 64-bit;
- macOS Apple Silicon;
- Linux x86_64.

Source execution is available on systems where Python 3.13 and the required dependencies can be installed.

---

## Source Code

The instructions below are intended for users who want to run or modify SARgate from source code.

SARgate is a Python application. To run it from source, you need:

- a local copy of the SARgate project folder;
- Python installed on your computer;
- a terminal;
- an isolated Python environment, created either with Python `venv` or with Conda/Miniconda;
- the project dependencies installed inside that environment.

> **Important**
>
> SARgate depends on RDKit. RDKit can sometimes be difficult to install in a plain pip-only environment on some systems. For this reason, the recommended installation method is a Conda environment. A Python `venv` is still possible, but Conda is generally more reliable.

---

## Download the Source Code

First, place the SARgate source code on your computer.

If you downloaded the project as a ZIP archive, extract it to a folder of your choice.

If you are cloning the repository, run:

```bash
git clone https://github.com/I2DLab/SARgate.git
cd SARgate
```

The SARgate folder is the folder that contains files and folders such as:

```text
SARgate/
├── launcher.py
├── assets/
├── data/
├── installer/
└── app/
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

---

## Recommended Installation: Conda / Miniconda

If you do not already have Conda, install Miniconda first.

### Install Miniconda

Go to the official Miniconda website and download the installer for your operating system:

<https://docs.conda.io/en/latest/miniconda.html>

Install it with the default options unless your institution has specific policies.

After installation, close and reopen the terminal so that the `conda` command becomes available.

To verify that Conda is correctly installed, run:

```bash
conda --version
```

If the command prints a version number, Conda is ready.

### Create a Dedicated Environment

Create a new environment for SARgate. Python 3.13 is the safest target for this project.

```bash
conda create -n sargate python=3.13
```

When Conda asks for confirmation, type:

```text
y
```

### Activate the Environment

Activate the new environment:

```bash
conda activate sargate
```

Your terminal prompt should now show the environment name, usually as:

```text
(sargate)
```

### Install the Project Dependencies

From inside the SARgate project folder, install the dependencies listed in `installer/requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r installer/requirements.txt
```

If installation completes without errors, SARgate is ready to launch.

### If RDKit Fails to Install with pip

If the previous command fails because RDKit cannot be installed, use Conda packages instead.

Run:

```bash
conda install -c conda-forge rdkit
python -m pip install dearpygui networkx numpy openpyxl pandas Pillow plotly pynput reportlab requests scikit-learn scipy screeninfo
```

This mixed approach is often the most robust solution on many systems.

---

## Alternative Installation: Python venv

Use this method only if you prefer a standard Python virtual environment and if your platform can install all required packages successfully.

### Install Python

Download and install Python from:

<https://www.python.org/downloads/>

Recommended version:

```text
Python 3.13
```

On Windows, during installation, make sure to enable the option:

```text
Add Python to PATH
```

After installation, open a new terminal and verify that Python is available.

On macOS or Linux:

```bash
python3 --version
```

On Windows:

```bat
python --version
```

### Create the Virtual Environment

Move to the SARgate project folder, then create a virtual environment.

On macOS or Linux:

```bash
python3 -m venv .venv
```

On Windows:

```bat
python -m venv .venv
```

### Activate the Virtual Environment

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you may need to allow local scripts for the current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then close and reopen PowerShell, return to the SARgate folder, and activate the environment again.

### Install the Dependencies

After activation, install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r installer/requirements.txt
```

If RDKit fails to install in a `venv`, switch to the Conda method described above.

---

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

Or:

```bash
python3 app/main.py
```

However, `launcher.py` remains the recommended entry point for normal source-based execution.

---

## Starting SARgate Again

After the first installation, you do not need to reinstall the dependencies every time.

You only need to:

1. open a terminal;
2. move into the SARgate project folder;
3. activate the same environment;
4. run `launcher.py`.

Example with Conda:

```bash
cd SARgate
conda activate sargate
python launcher.py
```

Example with `venv` on macOS or Linux:

```bash
cd SARgate
source .venv/bin/activate
python3 launcher.py
```

Example with `venv` on Windows:

```bat
cd SARgate
.venv\Scripts\activate.bat
python launcher.py
```

---

## Project Resources

The application expects its internal resources to remain in the project structure.

In particular:

| Resource Type | Location |
| --- | --- |
| Configuration files | `assets/config/` |
| Interface assets | `assets/` |
| Input and output working folders | `data/` |
| Source code modules | `app/` |

Do not move these folders unless you also update the code accordingly.

---

## Repository Structure

The project is organized around the following main folders and files:

```text
SARgate/
├── app/
│   └── main.py
├── assets/
│   └── config/
├── data/
├── installer/
│   ├── installer.py
│   ├── requirements.txt
│   └── SARgate.spec
├── launcher.py
├── README.md
└── LICENSE
```

---

## Common Problems and Solutions

### `python`, `python3`, or `conda` is not recognized

The corresponding software is either not installed or not available in `PATH`.

Reinstall it and reopen the terminal.

### Environment activation fails

Make sure that:

- you are inside the SARgate project folder;
- the environment was actually created;
- you are using the correct activation command for your operating system and shell.

### Dependency installation fails

First upgrade pip:

```bash
python -m pip install --upgrade pip
```

Then retry:

```bash
python -m pip install -r installer/requirements.txt
```

If RDKit fails specifically, use the Conda method.

### The GUI does not start

If the GUI does not start but the installation completed successfully, try launching:

```bash
python app/main.py
```

This can help determine whether the issue is in the splash launcher or in the main application startup.

### macOS asks for permissions

If you are on macOS and the system asks for permissions related to files, screen access, or application control, grant them if needed for normal execution.

---

## Updating the Environment After Source Code Changes

If the SARgate source code changes and the new version introduces additional dependencies, reactivate the environment and run again:

```bash
python -m pip install -r installer/requirements.txt
```

This ensures that the environment remains synchronized with the current project version.

---

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

---

## Building the Executable

If you want to create a packaged executable from the source code, SARgate includes a dedicated `installer/` folder with:

```text
installer/installer.py
installer/SARgate.spec
installer/requirements.txt
```

Before creating the executable, make sure that:

1. you are inside the SARgate project root folder;
2. your Python or Conda environment is activated;
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

Or, if your system uses `python3`:

```bash
python3 installer/installer.py
```

This script automatically switches to the project root, collects the `assets/` and `data/` folders, and runs PyInstaller with the correct arguments.

### Build Using the `.spec` File

If you prefer to build directly from the PyInstaller specification file, run:

```bash
pyinstaller installer/SARgate.spec
```

Or:

```bash
python -m PyInstaller installer/SARgate.spec
```

This method uses the explicit build definition contained in `installer/SARgate.spec`.

### Build Output

After a successful build, PyInstaller usually creates:

```text
build/
dist/
```

The exact format depends on the operating system:

- on macOS, you may obtain a `SARgate.app` bundle and/or a folder inside `dist/`;
- on Windows, you typically obtain a SARgate executable inside `dist/`;
- on Linux, you typically obtain an executable inside `dist/`.

### If the Build Fails

If packaging fails, check the following points:

- PyInstaller is installed in the currently active environment;
- all SARgate runtime dependencies are installed first from `installer/requirements.txt`;
- you are launching the build command from the SARgate project root folder;
- the `assets/`, `data/`, and `app/` folders are present and not renamed.

If needed, reinstall the dependencies and then retry:

```bash
python -m pip install -r installer/requirements.txt
python -m pip install pyinstaller
python installer/installer.py
```

---

## Third-party Software

SARgate depends on RDKit.

RDKit is distributed under its own license terms. SARgate's MIT license applies to SARgate source code, while third-party dependencies remain governed by their respective licenses.

For RDKit license information, refer to the RDKit project:

<https://www.rdkit.org/>

---

## Contributing

Contributions, bug reports, and feature requests are welcome through GitHub Issues.

Before opening an issue, please include:

- your operating system;
- whether you are using an executable build or the source version;
- the command you used to launch SARgate;
- the full error message, if available.

---

## Citation

If you use SARgate in academic work, please cite this repository.


```text
citation
```

---

## License

SARgate is released under the MIT License.

See the root-level [`LICENSE`](LICENSE) file for the full license text.
