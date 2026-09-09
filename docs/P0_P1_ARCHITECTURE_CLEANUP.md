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

## Remaining follow-up
Cooperative cancellation inside heavy stages, stronger cache keys, full endpoint reprojection residuals, richer Qt integration tests, and final removal of legacy render shims remain P2/P3 work.
