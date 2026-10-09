"""Batch guidance audit: run on real photos, output structured results."""
import sys, os, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pose_detector import PoseDetector
from core.orientation import analyze_orientation
from core.action_classifier import classify_action
from core.camera_analyzer import analyze_camera
from core.composition import analyze_composition
from core.guidance import generate_guidance, GuidanceState, Direction
from core.image_io import load_image

import cv2

def resize(image, max_side=1600):
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    s = max_side / max(h, w)
    return cv2.resize(image, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def run_batch(image_dir, output_csv, limit=0):
    det = PoseDetector()
    files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ])
    if limit > 0:
        files = files[:limit]

    rows = []
    t0 = time.time()
    no_person = 0
    errors = 0

    for i, fname in enumerate(files):
        path = os.path.join(image_dir, fname)
        img = load_image(path)
        if img is None:
            rows.append([fname, "ERROR", "", "", "", "", "", "", ""])
            errors += 1
            continue

        analysis = resize(img)
        pose = det.detect(analysis)
        if pose is None:
            rows.append([fname, "NO_PERSON", "", "", "", "", "", "", ""])
            no_person += 1
            continue

        orientation = analyze_orientation(pose)
        action = classify_action(pose)
        camera = analyze_camera(pose, analysis)
        composition = analyze_composition(analysis, pose)
        guidance = generate_guidance(action, orientation, camera, composition)

        state = guidance.state.value
        primary_cue = guidance.primary.cue if guidance.primary else ""
        primary_dir = guidance.primary.direction.value if guidance.primary else ""
        primary_target = guidance.primary.target if guidance.primary else ""
        secondary_cue = guidance.secondary.cue if guidance.secondary else ""
        facing = orientation.facing.value
        action_cat = action.category.value
        px, py = composition.subject_position
        knee_diff = action.features.get("knee_angle_diff", 0)
        stance = action.features.get("stance_width", 0)
        elbow_l = action.joint_angles.get("left_elbow", 180)
        elbow_r = action.joint_angles.get("right_elbow", 180)

        rows.append([
            fname, state, primary_cue, primary_dir, primary_target,
            secondary_cue, facing, action_cat,
            f"pos=({px:.2f},{py:.2f}) knee_diff={knee_diff:.0f} stance={stance:.2f} elbows=({elbow_l:.0f},{elbow_r:.0f})"
        ])

        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(files)} ({elapsed:.1f}s)", flush=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "state", "primary_cue", "direction", "target",
                     "secondary_cue", "facing", "action", "metrics"])
        w.writerows(rows)

    elapsed = time.time() - t0
    print(f"\nDone: {len(files)} images in {elapsed:.1f}s")
    print(f"  NO_PERSON: {no_person}, ERRORS: {errors}")
    print(f"  Output: {output_csv}")

    states = [r[1] for r in rows if r[1] not in ("ERROR", "NO_PERSON")]
    print(f"\n  State distribution:")
    for s in ["adjust", "hold", "insufficient_evidence"]:
        count = states.count(s)
        print(f"    {s}: {count} ({count/max(len(states),1)*100:.0f}%)")

    dirs = [r[3] for r in rows if r[3]]
    print(f"\n  Direction distribution:")
    for d in sorted(set(dirs)):
        count = dirs.count(d)
        print(f"    {d}: {count}")

    targets = [r[4] for r in rows if r[4]]
    print(f"\n  Primary target distribution:")
    for t in sorted(set(targets)):
        count = targets.count(t)
        print(f"    {t}: {count}")

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_batch(r"D:\Test_area\Image\train", "guidance_audit_train.csv", limit)
