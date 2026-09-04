# Portrait Image Breakdown

Photography Analysis & Reverse Engineering System

Based on YOLOv26 Pose, this system analyzes a portrait photograph to reverse-engineer shooting parameters, composition intent, and photographic techniques.

## Two-Phase Analysis Architecture

The system operates in two distinct phases for optimal responsiveness:

### Phase 1: Fast Analysis (automatic, ~1-2 seconds)

```
Image → PoseDetector → Orientation → ActionClassifier → CameraAnalyzer
      → CompositionAnalyzer → SuggestionEngine → UI
```

All 2D analysis completes quickly and populates the UI panels immediately:
- **Skeleton** — 17 COCO keypoints with confidence scores
- **Body Orientation** — facing direction, tilt, rotation angle
- **Action Category** — 15 pose types (standing, walking, squatting, etc.) with joint angles
- **Camera Analysis** — shot type, camera angle, subject ratio, dutch angle
- **Composition Analysis** — rule-of-thirds alignment, symmetry, headroom, balance
- **Suggestions** — next action recommendations, detailed tips, creative direction

### Phase 2: Full Reverse Engineering (background, non-blocking)

```
Image + Pose + Composition → ReverseEngineeringEngine → UI / Results / 3D Workspace
```

Runs in the background without blocking the UI:
- **Perspective Analysis** — vanishing points, line convergence, perspective strength
- **Camera Position** — height, distance, pitch, yaw, roll (with confidence ranges)
- **Focal Length Estimation** — wide/normal/telephoto classification + 35mm equivalent
- **Depth of Field** — DOF type, aperture estimation, foreground/background blur
- **Motion Blur** — blur type, direction, shutter speed estimation
- **Shooting Techniques** — 17 photography technique classifications
- **Camera Actions** — MOVE_FORWARD, ZOOM_IN, etc. with reasoning

### Performance Design

- Original high-resolution photos are used only for display
- Analysis runs on resized images (max 1600px side) for speed
- MonocularDepthProvider is lightweight (gradient-based, no deep learning)
- Simulation engine is disabled by default (use `enable_simulation=True` for full optimization)
- Results are cached by image hash — reopening the same image is instant

## Features

### Core Analysis (core/)
- **Skeleton Detection** — YOLOv26 Pose 17 keypoints + interpolated extension points
- **Body Orientation** — front/back/left/right + tilt angle
- **Action Recognition** — 15 action types with joint angle measurements
- **Camera Analysis** — shot type, camera angle, subject ratio
- **Composition Analysis** — rule-of-thirds, symmetry, leading lines, headroom
- **Next Action Suggestions** — intelligent recommendations based on current state

### Photography Reverse Engineering (reverse_engineering/)
- **Perspective Analysis** — vanishing point detection, line convergence
- **Camera Position Estimation** — height, distance, pitch, yaw, roll
- **Focal Length Estimation** — wide/normal/short_telephoto/telephoto + 35mm equivalent
- **Depth of Field Analysis** — foreground/background blur, aperture range
- **Motion Blur Analysis** — blur type, direction, shutter speed estimation
- **Shooting Technique Classification** — 17 photography techniques
- **Reverse Validation Engine** — virtual projection parameter optimization
- **Camera Action Suggestions** — MOVE_FORWARD/BACKWARD/ZOOM_IN with reasoning

### Output Format
All estimates include:
- Estimated value + possible range
- Confidence level (high/medium/low)
- Inference basis
- Uncertainty notes

## Running

```bash
# Install dependencies
pip install ultralytics opencv-python PySide6

# CLI analysis
python main.py --image photo.jpg --cli

# GUI
python main.py

# GUI with image pre-loaded
python main.py --image photo.jpg
```

### Smoke Test

```bash
# Basic tests (imports, API contracts, no display needed)
python test_smoke.py

# Full pipeline test with real image
python test_smoke.py --image dataset/HO9prKFboAAm5Q5.jpg
```

## GUI Layout

```
┌─────────────────────────────────────────────────────────────┐
│ [Open Image] [Dataset ▼] [Skeleton] [3x3 Grid] [Center] ...│
├──────────┬──────────────────────────┬───────────────────────┤
│ Analysis │     Image Canvas         │ Photography Insight   │
│          │                          │                       │
│ Skeleton │   ┌──────────────┐       │ Recommended Actions   │
│  90%     │   │              │       │  1. Walk forward      │
│  9/17    │   │   Photo +    │       │  2. Turn slightly     │
│          │   │   Overlays   │       │  3. Raise arm         │
│ Orient.  │   │              │       │                       │
│  Right   │   │  3x3 grid   │       │ Detailed Suggestions  │
│  Upright │   │  + skeleton  │       │  [HIGH] ...           │
│          │   │  + bbox      │       │  [MED] ...            │
│ Action   │   │  + center    │       │  [LOW] ...            │
│  Squat   │   │              │       │                       │
│  90%     │   └──────────────┘       │ Creative Direction    │
│          │                          │  ...                  │
│ Camera   │                          │                       │
│  Med-CU  │                          │                       │
│  Low ang │                          │                       │
│          │                          │                       │
│ Compose  │                          │                       │
│  Thirds  │                          │                       │
│  83%     │                          │                       │
├──────────┴──────────────────────────┴───────────────────────┤
│ Status: Core analysis complete | Squatting                  │
└─────────────────────────────────────────────────────────────┘
```

### Overlay Controls

| Toggle         | Description                                    |
|----------------|------------------------------------------------|
| Skeleton       | Body skeleton lines + joint keypoints          |
| 3x3 Grid       | Rule-of-thirds grid (always available)         |
| Center         | Image center crosshair                         |
| BBox           | Subject bounding box                           |
| Visual Weight  | Visual weight center marker                    |
| Headroom       | Headroom horizontal guide line                 |

### Tab Pages

| Tab                      | Content                                    |
|--------------------------|--------------------------------------------|
| 2D Analysis              | Fast analysis results + image with overlays |
| 3D Reverse Engineering   | RE visualization (future: OpenGL 3D)       |
| Results                  | Full reverse engineering text report        |

## Project Structure

```
photo/
├── main.py                     # Entry point (GUI / CLI)
├── test_smoke.py               # Smoke tests
├── README.md
├── core/
│   ├── pose_detector.py        # YOLOv26 skeleton detection
│   ├── orientation.py          # Body orientation analysis
│   ├── action_classifier.py    # Action category recognition
│   ├── camera_analyzer.py      # Camera position estimation
│   ├── composition.py          # Composition analysis
│   └── suggestion.py           # Next action suggestions
├── reverse_engineering/
│   ├── data_types.py           # Core data structures (EstimatedValue, Result types)
│   ├── perspective.py          # Perspective analysis
│   ├── camera_pose.py          # Camera position estimation
│   ├── focal_length.py         # Focal length estimation
│   ├── depth_of_field.py       # Depth of field analysis
│   ├── depth_provider.py       # Monocular depth estimation
│   ├── motion_blur.py          # Motion blur analysis
│   ├── shooting_technique.py   # Photography technique classification
│   ├── geometry.py             # Geometric constraint solver
│   ├── simulation.py           # Reverse validation engine
│   └── engine.py               # Main RE engine
├── gui/
│   ├── main_window.py          # PySide6 main window (staged worker, caching)
│   ├── canvas.py               # Image canvas + overlay rendering
│   └── panels.py               # Analysis & suggestion panels
└── dataset/                    # Sample images
```

## Data Structures

### AnalysisBundle (gui/main_window.py)
Accumulates all analysis results for a single image:
- `pose` — PoseResult (skeleton detection)
- `orientation` — OrientationResult (body facing)
- `action` — ActionResult (pose classification + joint angles)
- `camera` — CameraResult (shot type, angle, ratio)
- `composition` — CompositionResult (thirds, symmetry, balance)
- `suggestions` — SuggestionResult (next actions, tips, creative direction)
- `reverse_result` — ReverseEngineeringResult (full RE output)

### Worker Signals (AnalysisWorker)
- `pose_ready(PoseResult)` — skeleton detected, show immediately
- `core_ready(AnalysisBundle)` — all 2D analysis complete
- `reverse_ready(AnalysisBundle)` — full RE complete
- `error(str)` — error message

## Tech Stack

- Python 3.11+ (tested on 3.12)
- YOLOv26 (Ultralytics) — skeleton detection
- OpenCV 5.x — image processing
- PySide6 — GUI framework
- NumPy — numerical computation
