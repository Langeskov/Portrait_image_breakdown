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
- **Scene geometry constrains rotation** — Manhattan vanishing points, horizon direction and orthogonal image directions provide evidence for camera yaw/pitch/roll.
- **Focal length remains a candidate family** — a single image cannot generally determine exact focal length and distance independently.
- **Action labels provide context, not commands** — pose guidance is driven by body geometry, silhouette, balance, framing and visual intent.

## Intrinsics and Calibration

v2.5 includes reusable calibration profiles with explicit sensor, principal-point and pixel-aspect priors. EXIF metadata and user-selected calibration remain separate evidence sources.

## Evidence State Model

```text
OBSERVED
  ↓ directly supported by pixels / EXIF / selected calibration input

ESTIMATED
  ↓ inferred from geometry, pose, priors or model confidence

UNKNOWN
  ↓ insufficient evidence; do not treat as measured
```

## v2.5 Camera Geometry Constraints

The camera candidate pipeline now has a conservative support-plane constraint in addition to relative depth and broad height/distance feasibility intervals. A standing subject with reliable ankle/knee geometry can provide a weak contact-plane hypothesis. Its purpose is to rank pitch-coherent solutions, not to claim that a physical floor was definitively detected.

The application pitch convention is **positive = looking downward**. The 3D workspace exposes optical-axis miss from the subject target so a candidate whose recovered pitch does not actually aim toward the subject is visible rather than silently corrected.

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
└── reverse_engineering/
    ├── geometry.py
    ├── intrinsics.py
    ├── calibration.py
    ├── depth_provider.py
    ├── scene_constraints.py
    ├── support_plane.py
    ├── simulation.py
    └── engine_v2.py
```

## Roadmap

### v2.5 — Field Photography Assistance

#### Completed
- Reusable camera calibration profiles with sensor size, principal point and pixel aspect ratio
- Calibration profiles bound to the active camera-fitting and projection intrinsics path
- EXIF + calibration evidence separation
- GUI calibration profile selector and reconstruction refresh
- CLI `--calibration-profile` selection
- Photographer cue presentation modes: concise / normal / technical
- Explicit observed / estimated / unknown evidence states in serialized results and GUI Results view
- Regression coverage for calibration-driven projection intrinsics, cue modes and evidence-state semantics
- Relative monocular depth constraint used as a soft candidate-ranking signal
- Broad camera height/distance feasibility intervals
- Support-plane pitch coherence constraint for reliable lower-limb contact geometry
- 3D optical-axis aim-error diagnostic; recovered pitch is no longer hard-clamped by vertical scene-line detection
- Regression coverage for support-plane pitch convention and candidate coherence

#### Next
- Mature monocular depth provider behind the existing `DepthProvider` interface
- Stronger non-Manhattan scene handling
- Image-space refinement against the original photograph beyond pose/BBox evidence
- Better pose landmarks for hands, feet and facial direction
- One-screen field mode that keeps verbal instructions visible while preserving technical analysis
- Cue history and undo so the photographer can compare successive pose adjustments
- Voice-ready cue text as a device-independent output layer

### v3 — Reference Reconstruction and Scene Understanding
- Multi-person 3D layout when independent depth evidence exists
- Room/object plane reconstruction and editable scene anchors
- Camera-to-scene calibration workflow using manually selected reference points
- Reference-photo comparison mode for recreating a known shot
- Pose-to-reference delta analysis: show which body parts need to move and in which direction
- Composition-aware target pose generation rather than only corrective suggestions
- Temporal mode for video/live camera input, smoothing pose and camera estimates over time

### v4 — Assisted Shooting
- Optional live camera/tether integration
- Near-real-time pose feedback during shooting
- Voice output for photographer cues so the photographer does not need to look at the screen
- Session records: image, camera hypothesis, pose state, verbal cues and operator adjustments
- Offline-first model packaging and inference profiles for field machines without network access
