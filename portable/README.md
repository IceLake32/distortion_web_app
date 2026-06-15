# Portable Builds

This folder contains scripts for building no-install portable releases of the
Distortions Streamlit demo.

The portable release works like this:

1. The build script creates a folder containing the app source and a local
   Python virtual environment.
2. Dependencies are installed into that local environment.
3. The user downloads a zip, unzips it, and double-clicks the launcher.
4. The launcher starts Streamlit locally and opens the app in a browser.

The app still runs as a local web app at `http://127.0.0.1:8501`; the user does
not need to install Python or run `pip install`.

## Windows

Run from the repository root on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\portable\build_windows.ps1
```

Output:

```text
dist\DistortionsDemo_Windows.zip
```

Users should unzip it and double-click:

```text
Start Distortions Demo.bat
```

## macOS

Run from the repository root on macOS:

```bash
bash portable/build_macos.sh
```

Output:

```text
dist/DistortionsDemo_macOS.zip
```

Users should unzip it and double-click:

```text
Start Distortions Demo.command
```

If macOS blocks the script because it was downloaded from the internet, the user
can right-click the command file and choose Open.
