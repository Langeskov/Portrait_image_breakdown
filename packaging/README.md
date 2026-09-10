# Windows release build

The Windows release uses PyInstaller in onedir mode. The pose checkpoints are intentionally excluded from Git by `.gitignore`, so the release machine must have all five local `.pt` files in `model/`.

Expected files:

- `model/yolo26n-pose.pt`
- `model/yolo26s-pose.pt`
- `model/yolo26m-pose.pt`
- `model/yolo26l-pose.pt`
- `model/yolo26x-pose.pt`

The application icon is `assets/app.ico`; the old PNG icon is no longer used.

From PowerShell at the repository root:

```powershell
.\packaging\build_pyinstaller_windows.ps1
```

The script uses `packaging/portrait_image_breakdown.spec` and writes the distribution to:

```text
dist\PortraitImageBreakdown\
```

Run `PortraitImageBreakdown.exe` from that directory before wrapping the directory in an MSI or other installer. The distribution includes all local pose checkpoints and the `app.ico` asset. Packaged applications do not download missing built-in models.
