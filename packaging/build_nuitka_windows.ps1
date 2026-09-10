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

# Nuitka 4.0 had a known regression in the Torch package configuration around
# torch.utils._config_module. Use the current stable 4.2 series explicitly,
# and disable Torch JIT for standalone distribution as recommended by Nuitka.
$nuitkaSpec = "nuitka>=4.2,<4.3"

$pythonArgs = @(
    "run",
    "--with",
    $nuitkaSpec,
    "python",
    "-m",
    "nuitka",
    "--module-parameter=torch-disable-jit=yes",
    "--output-dir=$Root\build\nuitka",
    "--output-filename=PortraitImageBreakdown.exe",
    "$Root\main.py"
)

Write-Host "Building Portrait Image Breakdown with Nuitka ($nuitkaSpec)..."
Write-Host "Torch JIT: disabled for standalone distribution"
Write-Host "Models: $($requiredModels -join ', ')"

& uv @pythonArgs
if ($LASTEXITCODE -ne 0) {
    throw "Nuitka build failed with exit code $LASTEXITCODE"
}

$distDir = Join-Path $Root "build\nuitka\PortraitImageBreakdown.dist"
if (-not (Test-Path $distDir)) {
    throw "Nuitka completed but the expected distribution directory was not found: $distDir"
}

Write-Host ""
Write-Host "Standalone build ready:"
Write-Host $distDir
Write-Host ""
Write-Host "Run:"
Write-Host (Join-Path $distDir "PortraitImageBreakdown.exe")
