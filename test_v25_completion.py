"""v2.5 completion tests: refinement, anchors, history and voice output."""
import numpy as np

from core.cue_history import CueHistory
from core.landmark_quality import assess_landmarks, semantic_anchor_pixels
from core.photographer_cues import CuePriority, PhotographerCue
from core.voice_output import ssml, voice_ready_text
from reverse_engineering.geometry import CameraIntrinsics, PoseCandidate, _camera_pose_from_params
from reverse_engineering.image_refinement import subject_anchor


class LM:
    def __init__(self, x, y, visibility):
        self.x = x; self.y = y; self.visibility = visibility


def _kps():
    k = np.zeros((17, 3), dtype=float)
    for i in range(17):
        k[i] = (400 + (i % 5) * 25, 200 + i * 22, .9)
    k[11] = (450, 520, .95); k[12] = (550, 520, .95)
    return k


def test_subject_anchor_prefers_hips():
    k = _kps()
    a = subject_anchor(k)
    assert np.allclose(a, [500, 520])


def test_landmark_quality_and_semantic_anchor():
    lms = [LM(x, y, .9) for x, y, _ in _kps()]
    q = assess_landmarks(lms)
    assert q.visible_count == 17
    assert q.anchor_confidence > .8
    assert semantic_anchor_pixels(lms, 1000, 1000) == (500.0, 520.0)


def test_cue_history_undo_redo():
    def cue(text): return PhotographerCue(CuePriority.PRIMARY, text, "", "pose")
    h = CueHistory(capacity=3)
    h.push([cue("第一步")]); h.push([cue("第二步")])
    assert h.current.summary == "第二步"
    assert h.undo().summary == "第一步"
    assert h.redo().summary == "第二步"


def test_voice_output_is_device_independent():
    text = voice_ready_text("重心放到一条腿上。   看我。")
    assert text == "重心放到一条腿上。\n看我。"
    assert ssml(text).startswith("<speak>")


def test_refinement_handles_a_valid_candidate():
    from reverse_engineering.image_refinement import refine_camera_candidate
    k = _kps()
    intr = CameraIntrinsics.from_focal_mm(50, 1000, 800)
    _, extr = _camera_pose_from_params(4.0, 1.3, 0.0, 12.0, 0.0)
    candidate = PoseCandidate(intr, extr, 4.0, 1.3, 50.0, .7, {})
    refined = refine_camera_candidate(candidate, k, 1000, 800)
    assert np.isfinite(refined.extrinsics.position).all()
    assert abs(refined.extrinsics.pitch - 12.0) <= 9.0 + 1e-6
