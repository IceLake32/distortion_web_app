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

## Uploaded Data Format

The web app accepts CSV, TSV/TXT, XLSX, and XLS tables. Each row should be one
sample.

Users can either let the app compute the embedding or provide their own 2D
embedding columns.

For app-computed embeddings, the table needs at least two numeric feature
columns:

```text
feature_1,feature_2,feature_3,label
0.1,1.2,0.4,A
0.3,1.0,0.5,A
2.4,0.2,1.1,B
```

For uploaded embeddings, include two numeric embedding coordinate columns:

```text
feature_1,feature_2,feature_3,umap_1,umap_2,label
0.1,1.2,0.4,-2.1,0.5,A
0.3,1.0,0.5,-1.9,0.7,A
2.4,0.2,1.1,1.4,-0.3,B
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
