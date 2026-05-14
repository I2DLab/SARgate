SARgate
README

This guide is intended for users who want to run or modify the SARgate source code.
If you only want to use the program, you do not need to follow the steps below: simply launch the packaged executable distributed with the project (the macOS application bundle, the Windows .exe executable, or the Linux packaged executable, depending on your platform).


1. What You Need

SARgate is a Python application. To run it from source, you need:

- a local copy of the SARgate project folder;
- Python installed on your computer;
- a terminal;
- an isolated Python environment, created either with Python venv or with Conda/Miniconda;
- the project dependencies installed inside that environment.

Important note: SARgate depends on RDKit. RDKit can sometimes be difficult to install in a plain pip-only environment on some systems. For this reason, the recommended installation method is a Conda environment. A Python venv is still possible, but Conda is generally more reliable.


2. Download the Source Code

First, place the SARgate source code on your computer. If you received it as a ZIP archive, extract it to a folder of your choice. If you cloned it from a repository, make sure the full project folder is available locally.

In all the examples below, the SARgate folder is the folder that contains files such as:

- launcher.py
- assets/
- data/
- installer/
- app/

The main application entry point is now located at:

- app/main.py

The dependency file and packaging files are now stored inside the installer/ folder:

- installer/requirements.txt
- installer/installer.py
- installer/SARgate.spec

Open a terminal and move into that folder before doing anything else.

Example on macOS or Linux:

cd /full/path/to/SARgate

Example on Windows Command Prompt:

cd C:\full\path\to\SARgate

Example on Windows PowerShell:

cd "C:\full\path\to\SARgate"


3. Recommended Installation: Conda / Miniconda

If you do not already have Conda, install Miniconda first.

3.1. Install Miniconda

Go to the official Miniconda website and download the installer for your operating system:
https://docs.conda.io/en/latest/miniconda.html

Install it with the default options unless your institution has specific policies. After installation, close and reopen the terminal so that the conda command becomes available.

To verify that Conda is correctly installed, run:

conda --version

If the command prints a version number, Conda is ready.

3.2. Create a Dedicated Environment

Create a new environment for SARgate. Python 3.13 is the safest target for this project.

conda create -n sargate python=3.13

When Conda asks for confirmation, type:

y

3.3. Activate the Environment

Activate the new environment:

conda activate sargate

Your terminal prompt should now show the environment name, usually as:

(sargate)

3.4. Install the Project Dependencies

From inside the SARgate project folder, install the dependencies listed in installer/requirements.txt:

python -m pip install --upgrade pip
python -m pip install -r installer/requirements.txt

If installation completes without errors, SARgate is ready to launch.

3.5. If RDKit Fails to Install with pip

If the previous command fails because RDKit cannot be installed, use Conda packages instead.

Run:

conda install -c conda-forge rdkit
python -m pip install dearpygui networkx numpy openpyxl pandas Pillow plotly pynput reportlab requests scikit-learn scipy screeninfo

This mixed approach is often the most robust solution on many systems.


4. Alternative Installation: Python venv

Use this method only if you prefer a standard Python virtual environment and if your platform can install all required packages successfully.

4.1. Install Python

Download and install Python from:
https://www.python.org/downloads/

Recommended version: Python 3.13.

On Windows, during installation, make sure to enable the option:

Add Python to PATH

After installation, open a new terminal and verify that Python is available.

On macOS or Linux:

python3 --version

On Windows:

python --version

4.2. Create the Virtual Environment

Move to the SARgate project folder, then create a virtual environment.

On macOS or Linux:

python3 -m venv .venv

On Windows:

python -m venv .venv

4.3. Activate the Virtual Environment

On macOS or Linux:

source .venv/bin/activate

On Windows Command Prompt:

.venv\Scripts\activate.bat

On Windows PowerShell:

.venv\Scripts\Activate.ps1

If PowerShell blocks activation, you may need to allow local scripts for the current user:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Then close and reopen PowerShell, return to the SARgate folder, and activate the environment again.

4.4. Install the Dependencies

After activation, install the required packages:

python -m pip install --upgrade pip
python -m pip install -r installer/requirements.txt

If RDKit fails to install in a venv, switch to the Conda method described above.


5. Launch SARgate

Once the environment is active and the dependencies are installed, start SARgate from the project root folder.

Recommended command:

python launcher.py

The launcher shows the splash screen, preloads the major modules, and then starts the main GUI.

If your system uses python3 instead of python, use:

python3 launcher.py

You can also run the main application directly, without the splash screen:

python app/main.py

or:

python3 app/main.py

However, launcher.py remains the recommended entry point for normal source-based execution.


6. Every Time You Want to Start SARgate Again

After the first installation, you do not need to reinstall the dependencies every time. You only need to:

1. open a terminal;
2. move into the SARgate project folder;
3. activate the same environment;
4. run launcher.py.

Example with Conda:

cd /full/path/to/SARgate
conda activate sargate
python launcher.py

Example with venv on macOS or Linux:

cd /full/path/to/SARgate
source .venv/bin/activate
python3 launcher.py

Example with venv on Windows:

cd C:\full\path\to\SARgate
.venv\Scripts\activate.bat
python launcher.py


7. Where SARgate Stores Its Main Resources

The application expects its internal resources to remain in the project structure. In particular:

- configuration files are stored in assets/config/
- interface assets are stored in assets/
- input and output working folders are stored under data/
- source code modules are stored under app/

Do not move these folders unless you also update the code accordingly.


8. Common Problems and Solutions

If the terminal says that python, python3, or conda is not recognized, the corresponding software is either not installed or not available in PATH. Reinstall it and reopen the terminal.

If activating the environment fails, make sure you are inside the SARgate project folder and that the environment was actually created.

If dependency installation fails, first upgrade pip:

python -m pip install --upgrade pip

If RDKit fails specifically, use the Conda method.

If the GUI does not start but the installation completed successfully, try launching:

python app/main.py

This can help determine whether the issue is in the splash launcher or in the main application startup.

If you are on macOS and the system asks for permissions related to files, screen access, or application control, grant them if needed for normal execution.


9. Updating the Environment After Source Code Changes

If the SARgate source code changes and the new version introduces additional dependencies, reactivate the environment and run again:

python -m pip install -r installer/requirements.txt

This ensures that the environment remains synchronized with the current project version.


10. Recommended Minimal Workflow for Non-Programmers

If you are unfamiliar with programming and simply want the safest route to run the source code, follow this exact sequence:

1. Install Miniconda.
2. Open a terminal.
3. Move into the SARgate folder.
4. Run:

conda create -n sargate python=3.13
conda activate sargate
python -m pip install --upgrade pip
python -m pip install -r installer/requirements.txt
python launcher.py


11. How to Build the Executable

If you want to create a packaged executable from the source code, SARgate includes a dedicated installer folder with:

- installer/installer.py
- installer/SARgate.spec
- installer/requirements.txt

Before creating the executable, make sure that:

1. you are inside the SARgate project root folder;
2. your Python or Conda environment is activated;
3. PyInstaller is installed in that environment.

To install PyInstaller, run:

python -m pip install pyinstaller

11.1. Build Using installer.py

This is the simplest method. From the SARgate project root folder, run:

python installer/installer.py

or, if your system uses python3:

python3 installer/installer.py

This script automatically switches to the project root, collects the assets and data folders, and runs PyInstaller with the correct arguments.

11.2. Build Using the .spec File

If you prefer to build directly from the PyInstaller specification file, run:

pyinstaller installer/SARgate.spec

or:

python -m PyInstaller installer/SARgate.spec

This method uses the explicit build definition contained in installer/SARgate.spec.

11.3. Where the Build Output Will Appear

After a successful build, PyInstaller usually creates:

- a build/ folder with temporary build files;
- a dist/ folder containing the generated SARgate application bundle or executable.

The exact format depends on the operating system:

- on macOS, you may obtain a SARgate.app bundle and/or a folder inside dist/;
- on Windows, you typically obtain a SARgate executable inside dist/;
- on Linux, you typically obtain an executable inside dist/.

11.4. If the Build Fails

If packaging fails, check the following points:

- PyInstaller is installed in the currently active environment;
- all SARgate runtime dependencies are installed first from installer/requirements.txt;
- you are launching the build command from the SARgate project root folder;
- the assets/, data/, and app/ folders are present and not renamed.

If needed, reinstall the dependencies and then retry:

python -m pip install -r installer/requirements.txt
python -m pip install pyinstaller
python installer/installer.py
