SARgate - Instructions for executable releases
==============================================

This README is for users who download a pre-built SARgate executable.
If you downloaded the source code instead, use the dedicated source-code
README included with the repository.


Windows
-------

1. Download the Windows zip archive, for example:
   SARgate-win64.zip

2. Extract the zip archive.

3. Open the extracted SARgate folder and launch SARgate.exe.

Important:
SARgate.exe must remain in the same folder as the _internal folder.
Do not move SARgate.exe alone to the Desktop or to another directory.

Correct:
SARgate/
  SARgate.exe
  _internal/

Incorrect:
Desktop/
  SARgate.exe

You can place the whole SARgate folder wherever you prefer, for example in
Documents, Desktop, or another personal folder. If you want an icon on the
Desktop, create a shortcut/alias to SARgate.exe instead of moving the file.


Linux / Ubuntu
--------------

1. Download the Linux zip archive, for example:
   SARgate-linux-x86_64.zip

2. Extract the zip archive.

3. Open the extracted SARgate folder and launch the SARgate executable.

Important:
The SARgate executable must remain in the same folder as the _internal folder.
Do not move the executable alone to the Desktop or to another directory.

Correct:
SARgate/
  SARgate
  _internal/

Incorrect:
Desktop/
  SARgate

You can place the whole SARgate folder wherever you prefer. If you want a
launcher on the Desktop or in another location, create a link/shortcut to the
SARgate executable instead of moving it.

If Linux does not allow the executable to start, make sure it has execution
permission. From a terminal opened in the SARgate folder:

chmod +x ./SARgate


macOS
-----

1. Download the macOS zip archive, for example:
   SARgate-mac-arm64.zip

2. Extract the zip archive.

3. Move SARgate.app wherever you prefer, for example to Applications or keep it
   in Downloads.

4. Because this build is not notarized by Apple, macOS may report that the app
   is damaged or cannot be opened. In that case, remove the quarantine attribute
   once before opening the app.

How to remove quarantine:

1. Open Terminal.

2. Type the following command, followed by a space:

xattr -dr com.apple.quarantine

3. Drag SARgate.app from Finder into the Terminal window. macOS will paste the
   full path automatically.

4. Press Enter.

The final command should look like this, with your own user name/path:

xattr -dr com.apple.quarantine "/Users/yourname/Downloads/SARgate.app"

5. After the command finishes, double-click SARgate.app to open it.

Alternative macOS opening method:
Sometimes Ctrl-click or right-click on SARgate.app, then Open, then Open again
is enough. If macOS says that the app is damaged, use the xattr command above.


General notes
-------------

- Keep the extracted application folder intact on Windows and Linux.
- Do not rename or delete the _internal folder.
- Do not move only the executable out of its folder.
- On macOS, SARgate.app is self-contained, but macOS security may require the
  quarantine-removal step described above.
- SARgate may create user settings and log files in your operating system's
  standard user-data folder.
- If the application does not start, try extracting the zip archive again into
  a simple path without special characters.

