$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$requiredModels = @(
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
    "yolo26l-pose.pt",
    "yolo26x-pose.pt"
)

$missing = @(
    $requiredModels |
        Where-Object { -not (Test-Path (Join-Path $Root "model\$_")) }
)

if ($missing.Count -gt 0) {
    Write-Error ("Missing local pose models: " + ($missing -join ", ") + ". Download them into model\ before building the release.")
    exit 1
}

$pyinstallerSpec = "pyinstaller==6.22.2"
$distDir = Join-Path $Root "dist\PortraitImageBreakdown"
$iconFile = Join-Path $Root "assets\app.ico"
$modelDir = Join-Path $Root "model"
$assetDir = Join-Path $Root "assets"

if (-not (Test-Path $iconFile)) {
    throw "Application icon not found: $iconFile"
}

Write-Host "Building Portrait Image Breakdown with PyInstaller ($pyinstallerSpec)..."
Write-Host "Mode: onedir"
Write-Host "Icon: assets\app.ico"
Write-Host "Models: $($requiredModels -join ', ')"

# Build main.py directly instead of passing a .spec file so that the
# explicit --icon option is applied by PyInstaller itself.
$pyinstallerArgs = @(
    "run",
    "--with",
    $pyinstallerSpec,
    "python",
    "-m",
    "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onedir",
    "--windowed",
    "--name=PortraitImageBreakdown",
    "--distpath=$Root\dist",
    "--workpath=$Root\build\pyinstaller",
    "--icon=$iconFile",
    "--add-data=$modelDir;model",
    "--add-data=$assetDir;assets",
    "--collect-submodules=ultralytics",
    "$Root\main.py"
)

& uv @pyinstallerArgs
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    throw "PyInstaller build failed with exit code $exitCode"
}

if (-not (Test-Path (Join-Path $distDir "PortraitImageBreakdown.exe"))) {
    throw "PyInstaller completed but the expected executable was not found: $distDir\PortraitImageBreakdown.exe"
}

Write-Host ""
Write-Host "Standalone Windows build ready:"
Write-Host $distDir
Write-Host ""
Write-Host "Run:"
Write-Host (Join-Path $distDir "PortraitImageBreakdown.exe")
