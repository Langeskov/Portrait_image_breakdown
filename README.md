# Portrait Image Breakdown

Photography Analysis & Reverse Engineering System

A native PySide6 desktop tool for analyzing portrait photographs, exploring plausible camera configurations, and turning technical analysis into practical on-set guidance.

## v2 Architecture

v2 separates the two kinds of evidence that were previously mixed together:

```text
                         ┌─ 2D Pose / BBox ──→ framing evidence
Image → Analysis ────────┤
                         └─ Scene Lines / VP ─→ rotation evidence
                                      │
                                      ↓
                         focal + distance + height
                         + yaw + pitch + roll
                                      │
                                      ↓
                              ranked candidates
                                      │
                           3D scene / 2D projection
```

The important design rule is:

- **Human pose constrains framing** — subject scale, image position, body shape and plausible camera distance/height.
- **Scene geometry constrains rotation** when the image contains a reliable Manhattan structure.
- **Focal length remains a candidate family** — a single image cannot generally determine exact focal length and distance independently.
- **Weak/non-Manhattan scenes fall back to image-driven pose fitting and bounded refinement** instead of forcing a Manhattan solution.
- **Action labels provide context, not commands** — pose guidance is driven by body geometry, silhouette, balance, framing and visual intent.

## v2.5 Architecture

```text
Image
  │
  ├─ Pose + BBox ─────────────┐
  ├─ EXIF + calibration ──────┤
  ├─ Scene geometry ───────────┤
  └─ Relative / local depth ───┤
                               ↓
                      candidate camera family
                               ↓
                       constraint ranking
                               ↓
                    bounded image-space refine
                               ↓
                    3D proxy + 2D validation
                               ↓
               photographer cues / field mode
                               ↓
                    voice-ready text / SSML
```

### Intrinsics and Calibration

Reusable calibration profiles carry sensor size, principal point, pixel aspect ratio and optional focal priors. EXIF metadata and user-selected calibration remain separate evidence sources, while the active profile is applied to candidate projection intrinsics.

### Depth

`DepthProvider` remains the stable interface. The default provider is deterministic and offline-friendly; a local monocular-depth model can be injected through `model_fn` without changing the reconstruction pipeline. Depth remains a relative ranking signal rather than absolute metric depth.

### Image-space refinement

After candidate recovery, v2.5 makes only bounded corrections to yaw/pitch/roll. The refinement prefers a stable body anchor (hips → torso → head) and minimizes observed-vs-projected keypoint residuals while preserving the focal-length/distance ambiguity. The correction is reported rather than silently hidden.

### Non-Manhattan scenes

Scene-geometry fusion is gated by scene confidence and the availability of three reliable orthogonal directions. Weak or non-Manhattan scenes rely more heavily on pose framing and the bounded image-space refinement pass.

### Landmark quality

The field layer exposes landmark confidence, lower-body confidence, face confidence and a semantic anchor. The existing COCO 17-point detector remains backward compatible while higher-quality/extended landmark providers can be plugged in later.

## Evidence State Model

```text
OBSERVED
  ↓ directly supported by pixels / EXIF / selected calibration input

ESTIMATED
  ↓ inferred from geometry, pose, priors or model confidence

UNKNOWN
  ↓ insufficient evidence; do not treat as measured
```

## v2.5 Field Mode

Field Mode is a fourth workspace designed for shooting rather than post-analysis. The current layout uses a large primary cue, compact confidence/landmark status, secondary cues, undo/redo, and separate plain-text and SSML copy actions. The output layer is device-independent and does not require a network speech service.

## v3 Architecture

v3 changes the question from **“what camera probably made this image?”** to **“what do I need to change to reproduce this reference image?”**.

```text
Reference image ──→ reference pose + composition anchors ──┐
                                                            ├─→ target delta
Current image   ──→ current pose + composition  ────────────┘
                                                            │
                                                            ↓
                                             camera / framing / pose guidance
```

The first v3 layer is intentionally 2D-first. A single photograph does not provide enough evidence for arbitrary metric room reconstruction, so reference reconstruction uses explicit image-space anchors and reports deltas instead of inventing absolute scene coordinates.

### Reference Reconstruction

`reverse_engineering/reference_reconstruction.py` provides:

- explicit reference composition records and editable semantic anchors
- stable `hip_center → shoulder_center → head` anchor selection
- reference/current subject-center and scale comparison
- per-landmark pose deltas with directional photographer instructions
- deterministic serialization for later reference-session recording

The GUI now provides a **Reference** workspace where a reference photograph can be loaded independently. The current analyzed image is compared automatically, making the first v3 loop usable without changing the existing v2.5 reconstruction engine.

## Project Structure

```text
photo/
├── main.py
├── test_smoke.py
├── test_stage2.py
├── test_v2_rotation.py
├── test_v2_regression.py
├── test_pose_guidance.py
├── test_photographer_cues.py
├── test_v25_foundation.py
├── test_v25_evidence_state.py
├── test_v25_scene_constraints.py
├── test_v25_candidate_family.py
├── test_v25_support_plane.py
├── test_v25_depth_provider.py
├── test_v25_completion.py
├── test_v3_reference_reconstruction.py
└── reverse_engineering/
    ├── geometry.py
    ├── intrinsics.py
    ├── calibration.py
    ├── depth_provider.py
    ├── scene_constraints.py
    ├── support_plane.py
    ├── image_refinement.py
    ├── reference_reconstruction.py
    ├── simulation.py
    └── engine_v2.py
```

## Roadmap

### v2.5 — Field Photography Assistance

**Functionally complete.**

Completed: calibration profiles, EXIF + calibration separation, multi-candidate camera fitting, depth/feasibility/support-plane ranking, optical-axis diagnostics, bounded image-space refinement, non-Manhattan fallback, landmark-quality layer, Field Mode, cue history, and voice-ready output. Regression coverage is included for the core v2.5 path.

### v3 — Reference Reconstruction and Scene Understanding

#### Completed first tranche
- Reference-photo comparison workspace
- Explicit reference composition and semantic body anchors
- Pose-to-reference landmark deltas
- Composition center and subject-scale deltas
- Directional photographer instructions derived from reference deltas
- Regression coverage for reference anchors, deltas and composition comparison

#### Next
- Multi-person 3D layout when independent depth evidence exists
- Room/object plane reconstruction and editable scene anchors
- Camera-to-scene calibration workflow using manually selected reference points
- Reference-photo camera hypothesis comparison
- Pose-to-reference delta visualization directly on the 2D canvas
- Composition-aware target pose generation rather than only corrective suggestions
- Temporal mode for video/live camera input, smoothing pose and camera estimates over time

### v4 — Assisted Shooting
- Optional live camera/tether integration
- Near-real-time pose feedback during shooting
- Voice output for photographer cues so the photographer does not need to look at the screen
- Session records: image, camera hypothesis, pose state, verbal cues and operator adjustments
- Offline-first model packaging and inference profiles for field machines without network access
