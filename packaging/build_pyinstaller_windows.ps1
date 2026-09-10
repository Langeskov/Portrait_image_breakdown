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
$specFile = Join-Path $Root "packaging\portrait_image_breakdown.spec"
$distDir = Join-Path $Root "dist\PortraitImageBreakdown"
$iconFile = Join-Path $Root "assets\app.ico"

if (-not (Test-Path $iconFile)) {
    throw "Application icon not found: $iconFile"
}

Write-Host "Building Portrait Image Breakdown with PyInstaller ($pyinstallerSpec)..."
Write-Host "Mode: onedir"
Write-Host "Icon: assets\app.ico (configured in the .spec file)"
Write-Host "Models: $($requiredModels -join ', ')"

# The .spec file owns build options such as the Windows icon. PyInstaller
# ignores most command-line options when a spec file is supplied, so do not
# pass --icon here; the icon is configured by EXE(..., icon=...) in the spec.
& uv @(
    "run",
    "--with",
    $pyinstallerSpec,
    "python",
    "-m",
    "PyInstaller",
    "--noconfirm",
    "--clean",
    "--distpath=$Root\dist",
    "--workpath=$Root\build\pyinstaller",
    $specFile
)
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
