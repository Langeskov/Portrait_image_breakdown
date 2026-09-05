# Portrait Image Breakdown

Photography Analysis & Reverse Engineering System

A native PySide6 desktop tool for analyzing portrait photographs and exploring plausible camera configurations.

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

### Phase 2: Reverse Engineering

```text
Image + Pose + Composition
        ↓
ReverseEngineeringEngine
        ↓
Perspective + Camera Geometry + Candidate Solutions
        ↓
2D Evidence Overlay / 3D Reverse Workspace / Results
```

The reverse-engineering layer includes:
- Perspective analysis and observed line segments
- Vanishing-point evidence
- Camera height / distance / orientation estimates with uncertainty
- Focal-length family estimation
- Depth-of-field analysis
- Motion-blur analysis
- Shooting-technique classification
- Camera-action recommendations
- Multiple plausible camera solutions instead of a single forced answer

## Important Interpretation Rule

A single 2D photograph normally does not uniquely determine focal length, camera distance, sensor format, or camera height. The system therefore distinguishes:

- **Observed** — image measurements such as keypoints, line segments, and vanishing-point candidates
- **Estimated** — quantities inferred from geometry or heuristics
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

The first-stage 3D workspace is a lightweight native desktop viewer built with QPainter. It intentionally avoids introducing a second 3D framework while the scene/data model is being stabilized.

It displays:

```text
Ground Grid
Subject Proxy
Camera
Camera Direction
Camera Frustum
Candidate Cameras
```

Interaction:
- Drag with the mouse to orbit the 3D view
- Mouse wheel to zoom the view
- Select a candidate solution
- Edit camera distance, height, yaw, pitch, roll and focal length

The first-stage 3D viewer is a visualization/validation tool, not a photogrammetric reconstruction engine.

## Candidate Solution Model

Focal length and subject distance are treated as coupled variables. For a portrait with insufficient scene scale, the engine can keep several plausible combinations, for example:

```text
#1  50 mm / 2.9 m
#2  70 mm / 4.1 m
#3  85 mm / 5.0 m
```

The actual values and scores depend on the detected subject geometry and available constraints.

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
│   ├── camera_pose.py
│   ├── focal_length.py
│   ├── depth_of_field.py
│   ├── depth_provider.py
│   ├── motion_blur.py
│   ├── shooting_technique.py
│   ├── geometry.py
│   ├── simulation.py
│   ├── scene.py
│   └── engine.py
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
pip install ultralytics opencv-python PySide6

python main.py
python main.py --image path/to/photo.jpg
python main.py --image path/to/photo.jpg --cli
```

## Roadmap

### Completed
- 2D analysis workspace
- Light desktop UI
- Staged background analysis
- Reverse-engineering evidence overlay
- Geometry/candidate solution refactor
- First-stage 3D camera reconstruction viewer
- Candidate solution selection and parameter editing
- In-session LRU analysis cache

### Next
- 2D ↔ 3D projection synchronization
- Better intrinsics estimation from EXIF/calibration
- Scene/depth constraints for camera distance and height
- Mature monocular/stereo/LiDAR depth providers
- More rigorous projective-geometry optimization
