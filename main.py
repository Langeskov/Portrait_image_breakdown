"""Portrait Image Breakdown - explicit application entry points."""
from __future__ import annotations

# Nuitka release configuration. The source build remains unchanged when run
# normally with Python, while ``python -m nuitka main.py`` becomes the
# standalone Windows distribution used for packaging.
# Nuitka 4.x currently has a Torch package-config regression around
# ``torch.utils._config_module``. The Windows release script pins the
# compiler to Nuitka 2.8.10, which is a known-good combination for this app.
# nuitka-project: --mode=standalone
# nuitka-project: --enable-plugin=pyside6
# nuitka-project: --include-data-dir={MAIN_DIRECTORY}/model=model
# nuitka-project: --include-data-dir={MAIN_DIRECTORY}/assets=assets
# nuitka-project-if: {OS} == "Windows":
#     nuitka-project: --windows-console-mode=disable

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _strip_evidence_block(text: str) -> str: