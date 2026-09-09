# v3 completion and technical debt audit

## v3 route completed

The v3 route is now covered end-to-end at the product/API level:

1. Reference-photo composition and semantic body anchors.
2. Reference target planning and 2D target guidance.
3. Multi-person relative layout and shared-camera 3D projection.
4. Editable point/plane scene anchors with independent visibility.
5. Image-first anchor binding and non-destructive camera hypotheses.
6. Reference Camera Hypothesis from composition evidence.
7. Plane projection changed to compact 2D reference-line evidence.
8. Semantic Horizontal / Vertical / Free reference-line constraints.
9. Evidence-derived roll correction with Apply / Undo and projection-residual feedback.
10. Plane-aware positional constraints for editable scene placement.
11. Versioned, integrity-checked editable reconstruction sessions.
12. Composition-aware pose target generation in current-image space.
13. Temporal smoothing primitives for pose and camera states.
14. Desktop exposure of session actions, plane constraints, generated pose targets, and a temporal video workspace.

The implementation keeps the important evidence boundary intact: a reference image can drive composition and re-aim hypotheses, while scene planes and explicit image evidence are kept separate from unsupported metric claims.

## Audit: historical technical debt

### P0 — integration architecture is too implicit

`main.py` currently installs the v2 engine by monkey-patching `reverse_engineering.engine.ReverseEngineeringEngine`, then monkey-patches GUI workspace/result methods and dynamically installs additional workspaces. This works, but makes import order part of runtime behavior and makes unit testing harder.

**Recommendation:** introduce an explicit `ApplicationServices`/`EngineProvider` object and pass it into `MainWindow`. Remove monkey patches in favor of concrete dependency injection. This should be done as a dedicated refactor after v3 stabilization.

### P0 — duplicate 3D workspace layers

The repository contains `reverse_3d.py`, `reverse_3d_workspace.py`, `reverse_3d_reference.py`, `reverse_3d_reference_line.py`, `reverse_3d_v3.py`, plus small feature modules layered through replacement/install functions. The split was useful while iterating quickly, but there are now multiple rendering entry points and repeated coordinate/layout logic.

**Recommendation:** consolidate into one `Reverse3DWorkspace` implementation plus small widgets (`ProjectionPreview`, `ReferenceLineOverlay`, `CameraMatchControl`, `AnchorInspector`). Keep compatibility imports only in one deprecated shim.

### P0 — private GUI coupling

The v3 integration accesses fields such as `_preview`, `_roll`, `_distance`, `_view`, `_reference_line_calibration`, `_refresh_projection`, and `_sync_controls` across widget boundaries.

**Recommendation:** expose narrow public methods (`set_camera_value`, `refresh_projection`, `set_selected_anchor`, `set_reference_evidence`, `load_scene`) and stop reaching into private widget internals.

### P1 — duplicated scene-anchor concepts

`reverse_engineering/reference_reconstruction.py` defines a small image-space `SceneAnchor` while `reverse_engineering/scene_anchors.py` defines the 3D `SceneAnchor`. The names are semantically different, but the collision is a recurring source of import mistakes and makes API discovery harder.

**Recommendation:** rename the image-space type to `ReferenceImageAnchor` and keep `SceneAnchor` reserved for world-space geometry.

### P1 — SceneModel has duplicated primary-subject state

`SceneModel` stores both `subject` and `subjects`, with `__post_init__` maintaining consistency. That is convenient for backwards compatibility but creates two mutable sources of truth.

**Recommendation:** make `subjects` canonical and expose `primary_subject` as a property. Migrate old call sites gradually.

### P1 — camera semantics are overloaded

`SceneCamera.yaw/pitch` are orbit parameters around a subject target, while the reference-camera layer reports re-aim deltas. These are currently kept separate by convention, but the type system does not distinguish them.

**Recommendation:** split `CameraOrbit` and `CameraOpticalAim` into explicit structures, then compose them into `SceneCamera`. This avoids accidental future code that adds re-aim to orbit angles.

### P1 — calibration evidence persistence is still scene-local

Reference-line evidence is persisted on `SceneAnchor.image_points` and `reference_line_constraint`. This is sufficient for a single current image, but it cannot yet represent multiple observations of the same physical feature across several images.

**Recommendation:** introduce an `EvidenceObservation` record with `image_id`, `points`, `constraint`, confidence, provenance, and timestamp/session id. Keep anchors as geometry and observations as image evidence.

### P1 — temporal UI is intentionally conservative

The Temporal workspace currently provides video pose smoothing. The core API can smooth camera states too, but the desktop temporal worker does not run a full per-frame camera reconstruction, because doing so would make the UI path computationally heavy and would blur the evidence boundary.

**Recommendation:** add an optional sampled camera-solve mode with explicit status/progress, then fuse only when per-frame camera evidence is available. Do not infer camera motion from smoothed pose alone.

### P1 — session schema migration is not yet version-migrating

The session loader validates an exact schema version. This is safe, but future changes will require explicit migrations rather than graceful upgrades.

**Recommendation:** add `migrate_session(data, from_version, to_version)` and retain deterministic fixtures for every historical session version.

### P2 — analysis worker lifecycle

The main analysis path still uses `terminate()` when a new image is loaded while a worker is running. Forced thread termination can leave native model/runtime resources in an uncertain state.

**Recommendation:** replace termination with a cooperative cancellation flag and make each heavy stage check it.

### P2 — cache key is intentionally small

The GUI cache uses a reduced-image hash. That is fast, but collisions are theoretically possible and the cache does not encode calibration profile or analysis settings.

**Recommendation:** include full-image content hash plus relevant analysis configuration in the cache key. Keep the small hash only as an optional prefilter.

### P2 — line residual geometry is still 2D-only

The Apply/Undo residual compares the observed image line angle with the projected plane reference-line angle. It is useful for UI feedback, but it is not a full reprojection RMS and does not measure endpoint distance.

**Recommendation:** add line endpoint residual, perpendicular distance, and a normalized combined score before using the metric for automatic optimization.

### P2 — test suite is strong on deterministic primitives, weaker on GUI integration

The CI deliberately excludes the full image pipeline and current tests focus on deterministic geometry/serialization contracts. That is good for stability, but the v3 UI currently lacks screenshot-level or widget interaction tests.

**Recommendation:** add offscreen Qt tests for selecting anchors, placing line endpoints, applying/undoing roll, saving/loading sessions, and replacing the preview widget. Keep real detector/model tests separate.

### P3 — unused and compatibility modules

There are legacy modules (`engine.py` / `engine_v2.py`, multiple reverse-3D compatibility shims) that now mainly support historical imports.

**Recommendation:** mark compatibility modules explicitly, document the canonical entry points, then remove them only after one or two release cycles.

## Optimization opportunities

- Centralize image-to-widget / widget-to-image coordinate transforms in one reusable class; current overlays repeat this math.
- Centralize camera projection construction so all widgets use exactly the same intrinsics/extrinsics object.
- Replace ad-hoc `__import__` calls in GUI bootstrap code with normal imports after the dependency graph is stabilized.
- Move user-visible strings into a small UI text catalog so Chinese/English labeling does not become scattered through rendering code.
- Add structured logging around evidence changes, camera edits, session load/save, and temporal processing; keep full tracebacks in diagnostics but make routine state transitions searchable.
- Add deterministic sample scenes containing one ground plane, one wall plane, one reference line, and one point anchor for regression testing without needing a photograph.

## Recommended refactor order

1. Consolidate the 3D workspace and make public APIs for scene/camera/preview synchronization.
2. Replace `main.py` monkey-patching with dependency injection.
3. Separate world geometry (`SceneAnchor`) from image observations (`EvidenceObservation`).
4. Split camera orbit parameters from optical aim parameters.
5. Introduce session schema migrations and stronger cache keys.
6. Add offscreen Qt integration tests.
7. Optimize temporal camera solving only after the preceding contracts are stable.
