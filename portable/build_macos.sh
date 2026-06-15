#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_ROOT="$REPO_ROOT/build/portable_macos"
APP_ROOT="$BUILD_ROOT/DistortionsDemo"
DIST_ROOT="$REPO_ROOT/dist"
ZIP_PATH="$DIST_ROOT/DistortionsDemo_macOS.zip"
PYTHON="${PYTHON:-python3}"

rm -rf "$BUILD_ROOT"
mkdir -p "$APP_ROOT" "$DIST_ROOT"
rm -f "$ZIP_PATH"

cp -R "$REPO_ROOT/web_app" "$APP_ROOT/web_app"
cp -R "$REPO_ROOT/distortions" "$APP_ROOT/distortions"
cp "$REPO_ROOT/requirements.txt" "$APP_ROOT/requirements.txt"
cp "$REPO_ROOT/LICENSE" "$APP_ROOT/LICENSE"
cp "$REPO_ROOT/README.md" "$APP_ROOT/README.md"

cd "$APP_ROOT"
"$PYTHON" -m venv .venv
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements.txt

cat > "Start Distortions Demo.command" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
open "http://127.0.0.1:8501" >/dev/null 2>&1 || true
echo "Starting Distortions Demo..."
echo "Keep this window open while using the app."
".venv/bin/python" -m streamlit run "web_app/app.py" \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true \
  --server.fileWatcherType none
EOF

chmod +x "Start Distortions Demo.command"
cd "$BUILD_ROOT"
zip -r "$ZIP_PATH" "DistortionsDemo"
echo "Created $ZIP_PATH"
