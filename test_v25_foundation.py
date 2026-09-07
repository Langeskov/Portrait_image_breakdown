"""v2.5 foundation tests: calibration profiles, intrinsics and field cue modes."""
from pathlib import Path
import tempfile

from core.photographer_cues import CuePriority, PhotographerCue
from core.photographer_cue_modes import CueMode, format_cues, primary_cue
from reverse_engineering.calibration import (
    BUILTIN_PROFILES,
    CalibrationProfile,
    load_profile,
    save_profile,
)
from reverse_engineering.intrinsics import read_exif_intrinsics


def test_builtin_calibration_profiles_have_safe_defaults():
    generic = BUILTIN_PROFILES["Generic"]
    assert generic.pixel_aspect_ratio == 1.0
    assert generic.sensor_width_mm is None
    full_frame = BUILTIN_PROFILES["Full Frame 36x24"]
    assert full_frame.sensor_width_mm == 36.0
    assert full_frame.sensor_height_mm == 24.0


def test_profile_round_trip():
    profile = CalibrationProfile(
        name="Lab Camera",
        sensor_width_mm=35.8,
        sensor_height_mm=23.9,
        pixel_aspect_ratio=1.0,
        principal_point_x=1001.5,
        principal_point_y=752.0,
        default_focal_length_mm=50.0,
        note="test",
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "profile.json"
        save_profile(profile, path)
        loaded = load_profile(path)
    assert loaded == profile


def _cue(text: str, reason: str = "") -> PhotographerCue:
    return PhotographerCue(CuePriority.PRIMARY, text, reason, "pose")


def test_cue_modes_preserve_same_underlying_cue():
    cues = [_cue("重心放到一条腿上。", "释放对称感"), _cue("手肘别夹死。", "打开轮廓")]
    assert format_cues(cues, CueMode.CONCISE) == ["重心放到一条腿上。"]
    assert format_cues(cues, CueMode.NORMAL) == ["重心放到一条腿上。", "手肘别夹死。"]
    technical = format_cues(cues, CueMode.TECHNICAL)
    assert "原因：释放对称感" in technical[0]
    assert primary_cue(cues) == cues[0].cue


def test_exif_reader_can_apply_calibration_profile(tmp_path):
    from PIL import Image

    image_path = tmp_path / "fixture.jpg"
    Image.new("RGB", (2000, 1500), "white").save(image_path)
    profile = BUILTIN_PROFILES["Full Frame 36x24"]
    evidence = read_exif_intrinsics(image_path, profile=profile)
    assert evidence.calibration_profile == profile.name
    assert evidence.sensor_width_mm == 36.0
    assert evidence.sensor_height_mm == 24.0
    assert evidence.principal_point_x == 1000.0
    assert evidence.principal_point_y == 750.0
    assert evidence.pixel_aspect_ratio == 1.0
