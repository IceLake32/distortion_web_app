# Distortions Streamlit Demo

Streamlit demo for exploring local geometric distortions introduced by 2D
embeddings. The app is based on the local `distortions` package from the paper
repository and includes only the package source needed for deployment.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run web_app/app.py
```

## Streamlit Community Cloud

Use this repository and set the app entrypoint to:

```text
web_app/app.py
```

## Portable No-Install Release

For users who want to analyze their own data locally, build a portable zip:

```powershell
powershell -ExecutionPolicy Bypass -File .\portable\build_windows.ps1
```

or on macOS:

```bash
bash portable/build_macos.sh
```

The generated zip contains the app, the local `distortions` package, and a
Python environment with all dependencies. Users unzip it and double-click the
launcher; the app opens in a local browser session.
