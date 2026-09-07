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

The project deliberately reuses mature components: Ultralytics/YOLO for pose detection, OpenCV for line detection and pinhole projection, NumPy for geometry, and SciPy for bounded numerical fitting where appropriate.

## Analysis Architecture

### Phase 1: Fast 2D Analysis

```text
Image → PoseDetector → Orientation → Action → Camera
      → Composition → Suggestions → Photographer Cues → 2D Workspace
```

Provides:
- Skeleton detection — YOLO Pose 17 keypoints
- Body orientation — facing direction, tilt, rotation angle
- Action recognition — pose categories and joint angles
- Camera analysis — shot type, camera angle, subject ratio
- Composition — rule-of-thirds, symmetry, headroom, balance
- Goal-oriented pose coaching — geometry-based corrections instead of action matching
- Photographer-ready verbal cues — short instructions suitable for on-set use

The professional analysis remains available alongside the verbal guidance. The two layers serve different purposes: analysis explains **why** a change is useful; a cue tells the photographer/subject **what to do next**.

### Phase 2: Reverse Engineering v2

```text
Image + Pose + BBox
        ↓
Scene Geometry ─────────────┐
  line detection             │
  orientation clustering     │
  vanishing points           │
  horizon                    ↓
                     Camera Candidate Fusion
Pose / framing ─────────────┤
                            ↓
                  Focal + Distance + Height
                  Yaw + Pitch + Roll
                            ↓
                  2D / 3D validation
```

The reverse-engineering layer includes:
- Perspective analysis and observed line segments
- Robust Manhattan-style scene geometry
- Multiple vanishing-point evidence rather than one global intersection median
- Scene-derived camera rotation candidates
- Pose/BBox-derived framing candidates
- Fusion and ranking of scene + pose candidates
- Focal-length family estimation with uncertainty
- Depth-of-field analysis
- Motion-blur analysis
- Shooting-technique classification
- Camera-action recommendations
- Multiple plausible camera solutions instead of a single forced answer

## Photographer Guidance

The system intentionally keeps two outputs alive at the same time:

```text
Technical analysis
  ↓
Why is this pose / composition useful?

Photographer cue
  ↓
What should I say to the subject right now?
```

This is important for field use. A photographer may be working without a computer next to the camera, while still wanting access later to the full geometric and compositional explanation.

The cue layer therefore translates analysis into speakable instructions without deleting the underlying data. Typical cues include:

- “重心放到一条腿上，另一条腿放松一点。”
- “一只手离开身体一点，肘部别夹死。”
- “两条腿前后错一点，别让膝盖和脚踝落在同一条线上。”
- “动作往你的右边打开一点。”
- “身体先不动，下巴往一侧偏一点，看我。”

A cue is a suggestion, not a mandatory target. The system should prefer small, achievable corrections over forcing a named action.

## Action Guidance Philosophy

The system intentionally separates **classification** from **direction**.

```text
What is happening now?
        ↓
   Action classifier

How can this frame be improved?
        ↓
 Body geometry + silhouette + balance
        + orientation + composition + camera
        ↓
 Goal-oriented pose guidance
        ↓
 Photographer-ready verbal cues
```

For example, a result should not stop at `站立 → 转身`. It can instead produce a physical adjustment, an explanation, and a concise spoken command. This keeps action recognition useful without allowing the classifier to dictate the pose.

## Important Interpretation Rule

A single 2D photograph normally does not uniquely determine focal length, camera distance, sensor format, or camera height. Camera rotation also requires scene evidence; a human pose alone is not enough to establish absolute world orientation.

The system therefore distinguishes:

- **Observed** — image measurements such as keypoints, BBox, line segments and vanishing-point candidates
- **Estimated** — quantities inferred from geometry or constrained fitting
- **Unknown** — quantities that remain under-constrained

Candidate solutions are displayed explicitly in the 3D workspace rather than presenting one candidate as ground truth.

## 2D Workspace

The 2D workspace is the evidence view. It focuses on the original photograph and analysis results without mixing in 3D controls.

Overlay controls include:

| Overlay | Purpose |
|---|---|
| Skeleton | Pose skeleton and keypoints |
| 3x3 Grid | Rule-of-thirds reference |
| Center | Image center crosshair |
| BBox | Subject bounding box |
| Visual Weight | Visual-weight center |
| Headroom | Headroom guide |
| Reverse Evidence | Perspective lines, vanishing points, estimated camera axis |

Reverse Evidence is a single explicit toggle so the 2D view remains readable.

## 3D Reverse Engineering Workspace

The 3D workspace is a lightweight native desktop viewer built with QPainter. It intentionally avoids introducing another 3D framework while the scene/data model is being stabilized.

It displays:

```text
Ground Grid
Subject Proxy
Camera Body + Lens
Optical Axis
Frustum / Field of View
Subject Target
Candidate Camera Solutions
2D Projection Preview
```

Interaction:
- Drag with the mouse to orbit the 3D view
- Mouse wheel to zoom the view
- Select a candidate solution
- Edit camera distance, height, yaw, pitch, roll and focal length

The viewer is a geometric explanation/validation view, not a claim of full photogrammetric reconstruction.

## Candidate Solution Model

v2 ranks candidates using two separate evidence families:

```text
Pose / BBox evidence
        ↓
Framing quality

Scene geometry evidence
        ↓
Rotation + focal prior

Both
 ↓
Combined candidate score
```

The system deliberately retains several solutions because focal length and distance are coupled in a monocular image.

## Scene Geometry and Rotation

For scenes containing architectural or other approximately orthogonal structure, v2 detects line segments, clusters them by image orientation and estimates up to three Manhattan vanishing directions.

The rotation solver then:

1. Builds normalized camera rays from the vanishing points using the existing pinhole intrinsics model.
2. Uses orthogonality between vanishing directions as a focal-length constraint.
3. Builds an orthonormal world-to-camera rotation from the three directions.
4. Converts that rotation into the application's Yaw / Pitch / Roll convention.
5. Fuses the scene rotation candidate with pose-derived framing candidates.

When a photograph does not contain enough reliable orthogonal scene structure, v2 does not manufacture a confident absolute rotation.

## Intrinsics Evidence

v2 now accepts an optional `IntrinsicsEvidence` record. When the original image path is available, the CLI reads standard EXIF fields such as:

```text
FocalLength
FocalLengthIn35mmFilm
Make / Model
LensModel
```

The evidence is used as a **prior**, not blindly treated as ground truth. Neighboring focal candidates are retained because an exported/cropped image can preserve EXIF while changing the relationship between pixels and the original camera framing.

When EXIF does not contain a focal field, the pipeline continues using its geometry/framing candidate family without requiring metadata.

The evidence source is persisted in `ReverseEngineeringResult.intrinsics_evidence` and included in CLI reports/JSON output.

## Projection Validation

The 3D projection preview uses the same pose-driven 17-keypoint proxy that participates in camera fitting. Candidate camera position and rotation share one target-centered coordinate frame, so the camera drawn in 3D and the camera used for 2D projection are now the same geometric state.

The preview reports:

```text
3D projected subject size
bbox IoU against observed BBox
center displacement
```

This makes the 3D view a validation tool rather than a decorative frustum renderer.

## Performance and Cache

The GUI uses staged background analysis with visible progress feedback:

```text
Pose ready → 2D ready → Reverse ready
```

Analysis uses resized images for expensive processing. Results are cached in-session by an exact SHA-256 image key and stored in a small LRU cache, so switching between 2D / 3D / Results does not rerun inference.

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
├── README.md
├── core/
│   ├── pose_detector.py
│   ├── orientation.py
│   ├── action_classifier.py
│   ├── camera_analyzer.py
│   ├── composition.py
│   ├── suggestion.py
│   └── photographer_cues.py
├── reverse_engineering/
│   ├── data_types.py
│   ├── perspective.py
│   ├── scene_geometry.py
│   ├── rotation_solver.py
│   ├── intrinsics.py
│   ├── camera_pose.py
│   ├── focal_length.py
│   ├── depth_of_field.py
│   ├── depth_provider.py
│   ├── motion_blur.py
│   ├── shooting_technique.py
│   ├── geometry.py
│   ├── simulation.py
│   ├── scene.py
│   ├── engine.py
│   └── engine_v2.py
├── gui/
│   ├── main_window.py
│   ├── canvas.py
│   ├── reverse_3d.py
│   ├── cache.py
│   └── panels.py
└── dataset/
```

## Running

```bash
pip install -e .

python main.py
python main.py --image path/to/photo.jpg
python main.py --image path/to/photo.jpg --cli
```

The application entry point installs the active reverse-engineering engine for both GUI and CLI.

## Roadmap

### v2 Completed
- 2D analysis workspace
- Light desktop UI
- Staged background analysis with progress feedback
- Reverse-engineering evidence overlay
- Standard pinhole projection model
- Pose/BBox framing candidate generation
- Corrected pinhole distance initialization for portrait-scale subjects
- Manhattan scene geometry extraction
- Vanishing-point rotation recovery
- Scene + pose candidate fusion
- 2D ↔ 3D projection synchronization
- Native 3D camera visualization
- Pose-driven 3D subject proxy for projection validation
- Multi-person 2D pose display
- In-session LRU analysis cache
- EXIF focal-length / camera metadata evidence and candidate prior
- Candidate visibility checks for unstable off-frame projections
- Goal-oriented pose guidance
- Photographer-ready verbal cue generation
- CI regression coverage for camera rotation, camera fitting and pose guidance

### v2.5 — Field Photography Assistance
- Camera-specific calibration profiles: principal point, pixel aspect ratio, sensor dimensions and lens presets
- Explicit observed / estimated / unknown confidence presentation in the GUI
- Scene/depth constraints for camera distance and height
- Depth-provider abstraction with an optional monocular depth provider first
- Stronger non-Manhattan scene handling
- Image-space refinement against the original photograph beyond pose/BBox evidence
- Better pose landmarks for hands, feet and facial direction
- On-set photographer cue modes: concise / normal / technical
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

### Out of Scope for the Current Stage
- Treating monocular reconstruction as metrically exact photogrammetry
- Requiring cloud services for core analysis
- Replacing professional photographic judgment with a single “best pose” score
