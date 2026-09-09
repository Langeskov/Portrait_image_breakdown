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
- **Roll is treated as an independent camera property** — human shoulder/body tilt is never used as a direct Dutch-angle measurement.
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

After candidate recovery, v2.5 makes only bounded corrections to yaw/pitch. Roll is intentionally frozen unless an independent scene-roll measurement is available, because human pose alone cannot distinguish camera roll from subject lean. Any independent roll evidence is reported with the candidate rather than silently hidden.

### Camera roll hardening

Vanishing-point fitting can produce mathematically valid Euler rotations from the wrong line families. The rotation solver now estimates roll separately from raw line orientations and requires agreement between orthogonal scene families before accepting a non-zero Dutch angle. Otherwise roll is explicitly neutralized to `0°`. This prevents diagonal architecture, hair, clothing, railings and other scene texture from becoming a false camera-roll reference.

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

Field Mode is a fourth workspace designed for shooting rather than post-analysis. It uses a large primary cue, compact confidence/landmark status, secondary cues, undo/redo, and separate plain-text and SSML copy actions. The output layer is device-independent and does not require a network speech service. The UI uses the same light theme as the rest of the desktop application, including explicit popup and disabled-state styling.

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

The GUI now provides a **Reference** workspace where a reference photograph can be loaded independently, with EXIF orientation normalized before pose analysis. The current analyzed image is compared automatically, making the first v3 loop usable without changing the existing v2.5 reconstruction engine.

### v3 Phase 2 — Reference target planning

The second tranche converts raw reference deltas into a reproducible shooting target. `reverse_engineering/reference_targets.py` separates the plan into:

- **framing actions** — subject scale plus whole-subject horizontal/vertical placement
- **pose actions** — the largest visible landmark deltas from the reference
- **priority order** — fix composition first, then use a small number of pose changes

The Reference workspace renders this target plan under the two-image comparison. It remains image-space and conservative: the system does not pretend that a single image uniquely determines physical room coordinates or a unique camera translation.

### v3 Phase 2.2 — 2D target guides

The current-photo 2D canvas now consumes the same reference target data and draws:

- a dashed **TARGET FRAME** showing where the reference subject should occupy the current frame
- a **TARGET CENTER** marker and arrow showing whole-subject composition movement
- up to five largest actionable landmark arrows from the current pose toward the reference-normalized targets
- a toolbar **Reference Target** toggle so the visual guidance can be hidden without disabling the reference comparison workspace

Target landmark coordinates are stored in the reference image's normalized coordinate system, so arrows remain stable when reference and current photographs have different resolutions or aspect ratios.

### v3 Phase 2.3 — Multi-person layout and relative 3D

The `reverse_engineering/multi_person_layout.py` model turns detector multi-person output into an explicit scene-layout record. Each detected person receives normalized framing geometry, a torso/hip depth sample, and a normalized relative depth coordinate. The system distinguishes:

- **image-space only** when depth evidence is missing or weak
- **relative 3D ordering** when the local depth backend provides enough separation
- **metric camera/person distance** remains unknown without an independent scale source

The reconstruction workspace now projects every retained person through the same camera model. Primary and additional people are visually separated, and the 2D validation preview shows their independent projected bounds.

### v3 Phase 2.4 — Editable scene anchors

`reverse_engineering/scene_anchors.py` introduces an explicit, conservative scene scaffold for room/object reconstruction:

- point and plane anchors with editable world position, normal and size
- independent **enabled** and **visible** states, so presentation can be simplified without changing reconstruction participation
- deterministic validation and serialization
- a protected default **Ground plane** coordinate scaffold
- helper construction of a plane from three manually supplied world points

`SceneModel` carries these anchors independently from observed image evidence. A fresh anchor starts with zero confidence and `manual` / `scene scaffold` provenance; later calibration stages can bind anchors to selected image points and promote only the supported geometry.

The reconstruction inspector is scrollable and deliberately sparse. **Scene people is collapsed by default**, Candidate solutions remains collapsed, and Scene anchor rows have per-anchor visibility switches. The visibility state is presentation-only and does not disable calibration participation.

### v3 Phase 2.5 — Image-first manual anchor calibration

The first part of Phase 2.5 is now implemented through **Anchor Calibration**:

- the source photograph is displayed directly beside the calibration controls
- clicking the image writes the next `P1…P4` image coordinate; numeric coordinates remain available for precision work
- bound point anchors and four-point plane anchors are drawn back onto the original image
- visible anchors are controlled independently from reconstruction participation
- all bound anchors can generate a non-destructive `CameraAnchorHypothesis` through PnP when sufficient correspondences exist
- the active `SceneCamera` is never silently replaced by the hypothesis

The remaining Phase 2.5 work is intentionally kept separate: reference-camera comparison, plane-aware object constraints, and save/load editable reconstruction sessions.

## Test Organization

Regression coverage is kept deterministic and avoids requiring YOLO weights or network services. The suite covers geometry/projection conventions, camera fitting, calibration/EXIF evidence, normalized orientation, relative depth, scene constraints, image refinement, semantic anchors, evidence states, photographer cues, voice output, scene rotation, Field Mode, reference target planning, multi-person layout and scene-anchor contracts. Model-backed end-to-end tests that require real model weights or photographs remain outside the deterministic contract suite.

## Roadmap

### v2.5 — Field Photography Assistance

**Functionally complete.**

Completed: calibration profiles, EXIF + calibration separation, multi-candidate camera fitting, depth/feasibility/support-plane ranking, optical-axis diagnostics, bounded image-space refinement, conservative camera-roll handling, non-Manhattan fallback, landmark-quality layer, Field Mode, cue history, voice-ready output, and explicit EXIF image-orientation normalization.

### v3 — Reference Reconstruction and Scene Understanding

#### Completed
- Reference-photo comparison workspace
- Explicit reference composition and semantic body anchors
- Pose-to-reference landmark deltas
- Composition center and subject-scale deltas
- Directional photographer instructions derived from reference deltas
- Reference target plan and conservative action ordering
- Orientation-normalized reference loading
- 2D target-frame / target-center / landmark guidance overlays
- Multi-person layout foundation with conservative relative depth
- Multi-person 3D scene rendering and shared-camera projection
- Editable point/plane scene-anchor scaffold
- Independent scene-anchor visibility controls
- Image-first manual anchor binding with original-image overlay
- Non-destructive camera hypothesis estimation from bound anchors
- Compact scrollable reconstruction inspector with collapsible Scene people / Candidate sections

#### Phase 2.5 — remaining
- Reference-photo camera hypothesis comparison against the anchored scene
- Plane-aware room/object constraints without silently converting hypotheses into observations
- Save/load editable reconstruction sessions

#### Later v3
- Composition-aware target pose generation rather than only corrective suggestions
- Temporal mode for video/live camera input, smoothing pose and camera estimates over time

### v4 — Assisted Shooting
- Optional live camera/tether integration
- Near-real-time pose feedback during shooting
- Voice output for photographer cues so the photographer does not need to look at the screen
- Session records: image, camera hypothesis, pose state, verbal cues and operator adjustments
- Offline-first model packaging and inference profiles for field machines without network access
