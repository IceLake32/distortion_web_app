$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$BuildRoot = Join-Path $RepoRoot "build\portable_windows"
$AppRoot = Join-Path $BuildRoot "DistortionsDemo"
$DistRoot = Join-Path $RepoRoot "dist"
$ZipPath = Join-Path $DistRoot "DistortionsDemo_Windows.zip"
$Python = "python"

if (Test-Path $BuildRoot) {
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}
if (!(Test-Path $DistRoot)) {
    New-Item -ItemType Directory -Path $DistRoot | Out-Null
}
if (Test-Path $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

New-Item -ItemType Directory -Path $AppRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $RepoRoot "web_app") -Destination (Join-Path $AppRoot "web_app") -Recurse
Copy-Item -LiteralPath (Join-Path $RepoRoot "distortions") -Destination (Join-Path $AppRoot "distortions") -Recurse
Copy-Item -LiteralPath (Join-Path $RepoRoot "requirements.txt") -Destination (Join-Path $AppRoot "requirements.txt")
Copy-Item -LiteralPath (Join-Path $RepoRoot "LICENSE") -Destination (Join-Path $AppRoot "LICENSE")
Copy-Item -LiteralPath (Join-Path $RepoRoot "README.md") -Destination (Join-Path $AppRoot "README.md")

Push-Location $AppRoot
try {
    & $Python -m venv .venv
    & ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
    & ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

    @'
@echo off
cd /d "%~dp0"
echo Starting Distortions Demo...
echo Keep this window open while using the app.
start "" http://127.0.0.1:8501
".venv\Scripts\python.exe" -m streamlit run "web_app\app.py" --server.address 127.0.0.1 --server.port 8501 --server.headless true --server.fileWatcherType none
echo.
echo Streamlit stopped. Press any key to close this window.
pause > nul
'@ | Set-Content -Path "Start Distortions Demo.bat" -Encoding ASCII
}
finally {
    Pop-Location
}

Compress-Archive -Path $AppRoot -DestinationPath $ZipPath -Force
Write-Host "Created $ZipPath"
