# Portrait Image Breakdown

Photography analysis and camera reverse-engineering for portrait work.

A native **PySide6** desktop application that combines 2D pose/composition analysis with conservative camera estimation, editable 3D scene reconstruction, and reference-image shooting guidance.

## What it does

```text
Reference image ─┐
                 ├─→ composition / pose deltas ─→ shooting target
Current image  ──┘
                        │
                        ├─→ 2D overlays
                        ├─→ camera candidate family
                        ├─→ editable 3D scene
                        ├─→ image-space anchors / PnP cross-check
                        └─→ reference camera hypothesis
```

The application deliberately reports **evidence, estimates, and unknowns separately**. A single photograph cannot uniquely determine focal length, camera distance, or metric room geometry, so the system keeps ambiguous quantities as candidate families or hypotheses instead of presenting them as facts.

## Current V3 workflow

1. Load a reference image and a current image.
2. Compare pose, subject scale, center, and framing.
3. Use the 2D target overlay to guide composition and pose changes.
4. Open **3D Reverse Engineering** to edit Camera and Scene anchors.
5. Bind image-space points or plane evidence to selected scene anchors.
6. Inspect the anchor-based PnP cross-check and **Reference Camera Hypothesis**.
7. Add horizontal / vertical / free reference-line constraints and optional roll correction.
8. Apply plane-aware positional constraints and inspect the live 2D projection.
9. Save or restore a versioned `.pibr.json` reconstruction session.
10. Use the Temporal workspace for conservative pose smoothing on sampled sequences.

The 3D inspector keeps **Camera** and **Scene anchors** near the top, while **Scene people** stays at the bottom. The live **2D Projection Preview** highlights the selected plane or point to keep image-space evidence tied to the scene model.

## Architecture

The repository separates application composition, GUI presentation, analysis, and reconstruction mathematics:

```text
main.py
  └─ gui.application_runtime
      ├─ gui.application_window
      ├─ gui.main_window
      ├─ gui.field_mode
      ├─ gui.reference_mode
      ├─ gui.anchor_calibration_dialog
      └─ gui.v3_completion

GUI / V3
  gui.reverse_3d_v3
      ├─ gui.reverse_3d_workspace   # 3D inspector + anchor/projection workspace
      ├─ gui.reverse_3d             # low-level 3D scene/projection rendering
      ├─ gui.reverse_3d_reference_line
      │   └─ reference-line / camera-match controls
      └─ gui.reference_line_calibration
          └─ interactive line evidence + roll correction

Reverse engineering
  reverse_engineering.engine
      └─ reverse_engineering.engine_v2   # current 2.5 engine implementation
          ├─ camera / geometry / projection
          ├─ depth / support-plane evidence
          ├─ image-space refinement
          ├─ multi-person layout
          └─ shooting-technique scoring

Reference reconstruction
  reference_reconstruction
      └─ reference_anchor
  reference_camera
      ├─ reference_reconstruction
      ├─ reference_line_calibration
      └─ scene / scene_anchors
```

A more detailed dependency and cleanup map is kept in [`docs/DEPENDENCY_GRAPH.md`](docs/DEPENDENCY_GRAPH.md).

## Repository layout

```text
core/                  2D analysis, pose, composition, photographer cues
reverse_engineering/   camera geometry, reconstruction, reference reasoning
gui/                   desktop UI and V3 workspaces
tests/                 deterministic regression contracts
docs/                  architecture and domain notes
main.py                desktop / CLI entry point
pyproject.toml         Python package metadata and dependencies
uv.lock                locked environment
```

## Installation

Python **3.12+** is required.

```bash
uv sync
uv run python main.py
```

The deterministic test suite can be run with:

```bash
uv run pytest
```

Model-backed end-to-end checks may still require local YOLO weights and a suitable runtime environment; the regression suite is intentionally designed not to depend on network services.

## Evidence rules

- **Observed** — directly supported by pixels, EXIF, or explicit user input.
- **Estimated** — inferred from pose, geometry, priors, depth, or solver output.
- **Unknown** — insufficient evidence; it should not be treated as measured.

Important examples:

- focal length and distance are a candidate family, not a uniquely solved pair;
- relative monocular depth is used as a soft ranking signal, not metric room scale;
- camera roll is accepted only from independent scene-line evidence;
- reference-camera output is a delta/hypothesis and does not silently replace the active SceneCamera.

## Status

**V2.5:** functionally complete for the current deterministic camera-analysis pipeline.

**V3 Phase 2.5:** the current desktop reference-reconstruction loop is functionally complete, including reference targets, multi-person relative layout, editable scene anchors, image-space calibration, anchor PnP cross-checks, reference camera hypotheses, reference-line constraints, plane-aware constraints, reconstruction sessions, and temporal pose smoothing.

Future work focuses on richer whole-body target solving, multi-observation evidence records, sampled per-frame camera solving, and optional live camera/tether integration.
