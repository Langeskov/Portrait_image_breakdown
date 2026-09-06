# Portrait Image Breakdown

Photography Analysis & Reverse Engineering System

A native PySide6 desktop tool for analyzing portrait photographs and exploring plausible camera configurations.

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

The project deliberately reuses mature components: Ultralytics/YOLO for pose detection, OpenCV for line detection and pinhole projection, NumPy for geometry, and SciPy for bounded numerical fitting where appropriate.

## Analysis Architecture

### Phase 1: Fast 2D Analysis

```text
Image → PoseDetector → Orientation → Action → Camera
      → Composition → Suggestions → 2D Workspace
```

Provides:
- Skeleton detection — YOLO Pose 17 keypoints
- Body orientation — facing direction, tilt, rotation angle
- Action recognition — pose types and joint angles
- Camera analysis — shot type, camera angle, subject ratio
- Composition — rule-of-thirds, symmetry, headroom, balance
- Next-action suggestions

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

## Performance and Cache

The GUI uses staged background analysis:

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
├── README.md
├── core/
│   ├── pose_detector.py
│   ├── orientation.py
│   ├── action_classifier.py
│   ├── camera_analyzer.py
│   ├── composition.py
│   └── suggestion.py
├── reverse_engineering/
│   ├── data_types.py
│   ├── perspective.py
│   ├── scene_geometry.py
│   ├── rotation_solver.py
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

The application entry point installs `ReverseEngineeringEngineV2` as the active reconstruction engine for both GUI and CLI.

## Roadmap

### v2 Completed
- 2D analysis workspace
- Light desktop UI
- Staged background analysis
- Reverse-engineering evidence overlay
- Standard pinhole projection model
- Pose/BBox framing candidate generation
- Manhattan scene geometry extraction
- Vanishing-point rotation recovery
- Scene + pose candidate fusion
- 2D ↔ 3D projection synchronization
- Native 3D camera visualization
- Multi-person 2D pose display
- In-session LRU analysis cache
- CI regression coverage for camera rotation geometry

### Next
- Better intrinsics estimation from EXIF/calibration
- Scene/depth constraints for camera distance and height
- Mature monocular/stereo/LiDAR depth providers
- Multi-person 3D layout when independent depth evidence exists
- Stronger non-Manhattan scene handling
- Automatic refinement against the original image beyond pose/BBox evidence
