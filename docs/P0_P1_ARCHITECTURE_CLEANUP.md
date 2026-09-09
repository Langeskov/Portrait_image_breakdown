# P0/P1 architecture cleanup

## Canonical boundaries
- `ApplicationServices` owns engine and cache construction.
- `ApplicationMainWindow` is the application-owned injected window.
- `gui.reverse_3d_v3.Reverse3DWorkspace` is the canonical v3 reconstruction workspace.
- `SceneAnchor` is reserved for world-space scene geometry.
- `ReferenceImageAnchor` is reserved for image-space reference anchors.
- `SceneModel.subjects` is the canonical people collection; `subject` is compatibility access to the selected primary entry.
- reconstruction sessions use schema v2; v1 sessions are migrated on load.

## P0 cleanup
The application entry point composes concrete services and widgets explicitly. It does not rewrite module globals or class methods at runtime.

## P1 cleanup
World geometry and image observations use different names; primary-subject state is stored only once; sessions have an explicit migration path; camera reference deltas use separate reframe names rather than pretending to be orbit values.

## Compatibility
Legacy imports remain available where needed, but compatibility aliases are explicitly marked and are not used by the application composition layer.
