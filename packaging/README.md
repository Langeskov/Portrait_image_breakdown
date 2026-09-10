# Windows release build

The Windows release uses Nuitka standalone mode. The pose checkpoints are intentionally excluded from Git by `.gitignore`, so the release machine must have all five local `.pt` files in `model/`.

Expected files:

- `model/yolo26n-pose.pt`
- `model/yolo26s-pose.pt`
- `model/yolo26m-pose.pt`
- `model/yolo26l-pose.pt`
- `model/yolo26x-pose.pt`

From PowerShell at the repository root:

```powershell
.\packaging\build_nuitka_windows.ps1
```

The script uses the project options embedded in `main.py` and writes the standalone distribution to:

```text
build\nuitka\PortraitImageBreakdown.dist\
```

Run `PortraitImageBreakdown.exe` from that directory before wrapping the directory in an MSI. The distribution includes `model\` and `assets\` as external data directories and does not download missing built-in models when running as a packaged application.
