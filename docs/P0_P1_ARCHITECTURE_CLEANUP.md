# P0/P1 architecture cleanup

This cleanup establishes explicit application boundaries and removes duplicated mutable state while preserving the v3 evidence model.

## Canonical boundaries

- `ApplicationServices` owns engine/cache construction.
- `ApplicationMainWindow` is the injected application window.
- `gui.reverse_3d_v3.Reverse3DWorkspace` is the canonical v3 reconstruction workspace.
- `SceneAnchor` is reserved for world-space geometry.
- `ReferenceImageAnchor` is reserved for image-space reference anchors.
- `SceneModel.subjects` is the canonical people collection; `subject` is a compatibility accessor.
- Reconstruction sessions use schema v2 and migrate v1 deterministically.

## 2026-09 cleanup status

The historical GUI adapter layer has now been removed from the runtime path. `gui.reverse_3d_reference.py` and the standalone `gui.reference_line_apply.py` module were deleted after their remaining behavior was folded into the canonical V3 modules.

The 3D renderer/workspace split is intentionally retained: `gui.reverse_3d.py` provides low-level rendering widgets, while `gui.reverse_3d_workspace.py` provides the editable reconstruction workspace. `reverse_engineering.engine.py` remains the stable compatibility import while `engine_v2.py` carries the active V2.5 implementation.

See [`DEPENDENCY_GRAPH.md`](DEPENDENCY_GRAPH.md) for the current import graph and cleanup decisions.

## Remaining follow-up

Cooperative cancellation inside heavy stages, stronger cache keys, full endpoint reprojection residuals, richer Qt integration tests, and broader symbol-level dead-code analysis remain P2/P3 work.
