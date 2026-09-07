"""Unified regression suite for Portrait Image Breakdown."""
from __future__ import annotations

import cv2
import numpy as np

from core.action_classifier import ActionCategory, ActionResult
from core.camera_analyzer import CameraAngle, CameraResult, ShotType
from core.composition import CompositionResult, CompositionType
from core.cue_history import CueHistory
from core.image_io import _apply_exif_orientation, frame_orientation, load_image
from core.landmark_quality import assess_landmarks, semantic_anchor_pixels
from core.orientation import FacingDirection, OrientationResult, TiltDirection
from core.photographer_cue_modes import CueMode, format_cues, primary_cue
from core.photographer_cues import CuePriority, PhotographerCue, generate_photographer_cues, speakable_summary
from core.suggestion import generate_suggestions, _generate_pose_guidance
from core.voice_output import ssml, voice_ready_text
from reverse_engineering.calibration import BUILTIN_PROFILES, CalibrationProfile, load_profile, save_profile
from reverse_engineering.depth_provider import MonocularDepthProvider
from reverse_engineering.geometry import CameraIntrinsics, CameraModel, PoseCandidate, PoseSolver, _camera_pose_from_params, canonical_person_points
from reverse_engineering.image_refinement import subject_anchor, refine_camera_candidate
from reverse_engineering.intrinsics import read_exif_intrinsics
from reverse_engineering.reference_reconstruction import build_reference_composition, choose_semantic_anchor, compare_pose_to_reference, composition_delta
from reverse_engineering.reference_targets import build_reference_target_plan
from reverse_engineering.scene_constraints import DepthConstraintEvidence, CameraFeasibilityEvidence, build_depth_constraint_evidence, candidate_depth_score, candidate_feasibility_score
from reverse_engineering.scene_geometry import SceneGeometryEvidence, VanishingPoint
from reverse_engineering.scene import SceneModel, SceneCamera
from reverse_engineering.simulation import _dedupe_candidates, _intrinsics_from_profile
from reverse_engineering.support_plane import SupportPlaneEvidence, estimate_support_plane, expected_support_pitch_deg, candidate_support_plane_score


def _action(**features):
    base = {"knee_angle_avg": 160.0, "knee_angle_diff": 2.0, "stance_width": 0.05,
            "shoulder_y": 0.42, "hip_y": 0.43, "hands_above_shoulders": 0.0,
            "wrist_y_avg": 0.56, "ankle_y_diff": 0.02}
    base.update(features)
    return ActionResult(ActionCategory.STANDING, 0.85, "standard",
                        {"left_knee": 160.0, "right_knee": 160.0,
                         "left_elbow": 172.0, "right_elbow": 171.0,
                         "left_hip": 150.0, "right_hip": 150.0}, base)


def _orientation(facing=FacingDirection.FRONT, tilt=TiltDirection.UPRIGHT):
    return OrientationResult(facing, tilt, 0.0, 0.0, 0.9, "fixture")


def _camera(ratio=0.4):
    return CameraResult(ShotType.MEDIUM, CameraAngle.EYE_LEVEL, ratio,
                        (0.0, 0.0), 0.0, "fixture")


def _composition(x=0.5, y=0.5):
    return CompositionResult(CompositionType.CENTER, (x, y), 0.8, 0.5,
                             0.15, "balanced", (x, y), 0.8, [], "fixture")


def _kps():
    k = np.zeros((17, 3), dtype=float)
    for i in range(17):
        k[i] = (400 + (i % 5) * 25, 200 + i * 22, .9)
    k[11] = (450, 520, .95); k[12] = (550, 520, .95)
    return k


def _standing_pose():
    kp = np.zeros((17, 3), dtype=float); kp[:, 2] = 0.95
    kp[11, :2] = (400, 420); kp[12, :2] = (500, 420)
    kp[13, :2] = (410, 600); kp[14, :2] = (490, 600)
    kp[15, :2] = (410, 760); kp[16, :2] = (490, 760)
    for i in range(17):
        if i not in (11, 12, 13, 14, 15, 16): kp[i, :2] = (450 + (i % 2) * 5, 300 + i * 8)
    return kp


def _synthetic_reference_pose(width=1800, height=1200, focal=70.0, distance=6.0, camera_height=1.05):
    points = canonical_person_points()
    intr = CameraIntrinsics.from_focal_mm(focal, width, height)
    position = np.array([0.0, camera_height, -distance]); forward = -position; forward /= np.linalg.norm(forward)
    right = np.cross(np.array([0.0, 1.0, 0.0]), forward); right /= np.linalg.norm(right)
    up = np.cross(forward, right); up /= np.linalg.norm(up)
    rotation = np.vstack([right, -up, forward])
    rvec, _ = cv2.Rodrigues(rotation); tvec = -rotation @ position
    projected, _ = cv2.projectPoints(points, rvec, tvec, intr.to_matrix(), None)
    image_points = projected.reshape(-1, 2); kp = np.c_[image_points, np.ones(len(image_points))]
    bbox = tuple(np.round([image_points[:, 0].min(), image_points[:, 1].min(), image_points[:, 0].max(), image_points[:, 1].max()]).astype(int))
    return kp, bbox


def _reference_pose(offset_x=0.0, offset_y=0.0, scale=1.0):
    xy = np.array([[50,20],[45,18],[55,18],[40,20],[60,20],[35,45],[65,45],[28,65],[72,65],[25,82],[75,82],[40,80],[60,80],[42,112],[58,112],[42,145],[58,145]], dtype=float)
    xy[:, 0] = 50 + (xy[:, 0]-50)*scale + offset_x; xy[:, 1] = 20 + (xy[:, 1]-20)*scale + offset_y
    from core.pose_detector import PoseLandmark, PoseResult
    lms = [PoseLandmark(i, x, y, 0, .95, x/200, y/180, 0) for i,(x,y) in enumerate(xy)]
    return PoseResult(lms, 200, 180, .95, (int(xy[:,0].min()), int(xy[:,1].min()), int(xy[:,0].max()), int(xy[:,1].max())))


def test_public_pipeline_imports():
    import gui.main_window
    import gui.canvas
    import gui.field_mode
    import gui.reference_mode
    from reverse_engineering.engine import ReverseEngineeringEngine
    from reverse_engineering.engine_v2 import ReverseEngineeringEngineV2
    assert ReverseEngineeringEngine is ReverseEngineeringEngineV2
    assert ReverseEngineeringEngineV2.VERSION == "2.5"


def test_geometry_projection_contract():
    k = CameraIntrinsics.from_focal_mm(50.0, 4000, 3000)
    assert k.fx > 0 and 35 < k.fov_x < 45 and 25 < k.fov_y < 40
    ray = CameraModel(k).unproject_point(2000, 1500, 2.0)
    assert np.isfinite(ray).all() and abs(ray[2] - 2.0) < 1e-6
    cam = SceneCamera(distance=4.0, height=1.5); cam.pitch = 10.0
    assert np.isclose(np.linalg.norm(cam.forward()), 1.0) and cam.forward()[1] < 0


def test_proper_rotation_and_image_y_convention():
    _, extr = _camera_pose_from_params(5.0, 1.4, 12.0, -20.0, 6.0)
    R = cv2.Rodrigues(extr.rvec)[0]
    assert np.linalg.det(R) > 0.99 and np.allclose(R.T @ R, np.eye(3), atol=1e-6)
    intr = CameraIntrinsics.from_focal_mm(70.0, 2000, 1333); model = CameraModel(intr, extr)
    point = np.array([0.2, 0.4, 0.5]); px = np.asarray(model.project_point(point)); cp = R @ point + extr.tvec
    back = model.unproject_point(float(px[0]), float(px[1]), depth=float(cp[2]))
    recovered = np.linalg.inv(R) @ back + extr.position
    assert np.allclose(recovered, point, atol=1e-5)


def test_camera_fit_returns_ranked_family():
    kp, bbox = _synthetic_reference_pose()
    candidates = PoseSolver.fit_camera_to_pose(kp, 1800, 1200, subject_bbox=bbox, focal_seeds=(35,50,70,85,105), num_candidates=5)
    assert len(candidates) >= 3
    assert all(.5 <= c.distance <= 20 for c in candidates)
    assert abs(candidates[0].distance - 6.0) < 1.5
    assert candidates[0].losses["mean_reprojection_px"] < 20


def test_scene_model_projection_uses_pose_proxy():
    kp, bbox = _synthetic_reference_pose()
    from reverse_engineering.data_types import CameraPoseResult, CompositionResult as DC, DepthOfFieldResult, EstimatedValue, FocalLengthResult, MotionBlurResult, PerspectiveResult, ReverseEngineeringResult, ShootingTechniqueResult
    pose_landmarks = [type("LM", (), {"x": float(p[0]), "y": float(p[1]), "visibility": 1.0})() for p in kp]
    result = ReverseEngineeringResult((1800,1200), bbox, pose_landmarks, .3, [], {},
        PerspectiveResult(EstimatedValue(.2), EstimatedValue("normal"), [], EstimatedValue(0), EstimatedValue(0)),
        CameraPoseResult(EstimatedValue(1.05,"m"),EstimatedValue(6,"m"),EstimatedValue(0,"deg"),EstimatedValue(0,"deg"),EstimatedValue(0,"deg")),
        FocalLengthResult(EstimatedValue("normal", confidence=.8),EstimatedValue(70,"mm",confidence=.8)),
        DepthOfFieldResult(EstimatedValue("unknown"),True,0,0,EstimatedValue("unknown")),
        MotionBlurResult(EstimatedValue("none"),None,0,EstimatedValue("unknown")),
        DC([], (.5,.5), .3, .1, "balanced", .5), ShootingTechniqueResult([]), .5, [])
    scene = SceneModel.from_reverse_result(result); proxy = scene.subject.proxy_points()
    assert proxy.shape == (17,3) and np.isfinite(proxy).sum() > 30


def test_depth_provider_backends():
    image = np.zeros((80,100,3), dtype=np.uint8); image[20:60,30:70]=180
    p = MonocularDepthProvider(); d=p.estimate_relative_depth(image)
    assert d.shape==(80,100) and np.isfinite(d).all() and 0 <= float(d.min()) <= float(d.max()) <= 1 and p.last_backend == "structure_prior"
    expected=np.linspace(0,1,32*40,dtype=np.float32).reshape(32,40)
    local=MonocularDepthProvider(model_fn=lambda _: expected); assert np.allclose(local.estimate_depth(np.zeros((32,40,3),dtype=np.uint8)), expected); assert local.last_backend=="local_model"
    bad=MonocularDepthProvider(model_fn=lambda _: (_ for _ in ()).throw(RuntimeError("offline"))); assert bad.estimate_depth(np.zeros((32,40,3),dtype=np.uint8)).shape==(32,40); assert bad.last_backend=="structure_prior"


def test_depth_constraint_is_relative_and_conservative():
    from core.pose_detector import PoseLandmark, PoseResult
    lms=[PoseLandmark(i,300+(i%5)*40,180+i*28,0,.95,0,0,0) for i in range(17)]; pose=PoseResult(lms,800,800,.95,(260,150,520,700))
    image=np.zeros((800,800,3),dtype=np.uint8); image[:,:,0]=np.arange(800,dtype=np.uint8)[:,None]
    depth,e=build_depth_constraint_evidence(image,np.array([[lm.x,lm.y,lm.visibility] for lm in pose.landmarks]),MonocularDepthProvider())
    assert depth.shape==(800,800) and e.valid_count==17 and e.to_dict()["relative_only"] is True and 0 <= e.confidence <= .9
    weak=DepthConstraintEvidence(tuple([.5]*17),0,17,800,800); intr=CameraIntrinsics.from_focal_mm(50,800,800); _,extr=_camera_pose_from_params(4,1.4,0,0,0); c=PoseCandidate(intr,extr,4,1.4,50,.5,{})
    assert candidate_depth_score(c,np.array([[lm.x,lm.y,lm.visibility] for lm in pose.landmarks]),800,800,weak) is None


def test_calibration_profiles_and_exif(tmp_path):
    profile=CalibrationProfile("Lab Camera",35.8,23.9,1.0,1001.5,752.0,50.0,"test"); path=tmp_path/"profile.json"; save_profile(profile,path); assert load_profile(path)==profile
    ff=BUILTIN_PROFILES["Full Frame 36x24"]; assert ff.sensor_width_mm==36 and ff.sensor_height_mm==24
    image_path=tmp_path/"fixture.jpg"; cv2.imwrite(str(image_path),np.full((1500,2000,3),255,dtype=np.uint8)); evidence=read_exif_intrinsics(image_path,profile=ff)
    assert evidence.sensor_width_mm==36 and evidence.sensor_height_mm==24 and evidence.principal_point_x==1000 and evidence.principal_point_y==750
    full=_intrinsics_from_profile(50,2000,1500,ff); aps=_intrinsics_from_profile(50,2000,1500,BUILTIN_PROFILES["APS-C 23.5x15.6"]); assert aps.fx>full.fx and aps.fy>full.fy


def test_evidence_state_contract():
    from reverse_engineering.data_types import EstimatedValue, ConfidenceLevel
    observed=EstimatedValue(50,"mm",confidence=1,is_observed=True); estimated=EstimatedValue(50,"mm",confidence=.72); unknown=EstimatedValue(0,"mm",confidence=0)
    assert observed.evidence_state=="observed" and estimated.evidence_state=="estimated" and unknown.evidence_state=="unknown"
    assert observed.to_dict()["evidence_state"]=="observed" and unknown.to_dict()["confidence_level"]==ConfidenceLevel.UNKNOWN.value


def test_candidate_family_and_feasibility():
    def candidate(f,d,h,score):
        _,extr=_camera_pose_from_params(d,h,0,-8,0); return PoseCandidate(CameraIntrinsics.from_focal_mm(f,1600,1200),extr,d,h,f,score,{"mean_reprojection_px":10})
    ranked=_dedupe_candidates([candidate(50,3,1.3,.9),candidate(70,4.2,1.4,.82),candidate(85,5,1.5,.76)],3); assert len(ranked)==3 and [c.focal_equiv_35mm for c in ranked]==[50,70,85]
    evidence=CameraFeasibilityEvidence((2,5),(1,1.8),.7,("pose body extent",)); assert np.isclose(candidate_feasibility_score(candidate(70,3.5,1.4,.8),evidence),1)


def test_support_plane_pitch_contract():
    evidence=estimate_support_plane(_standing_pose(),900,800); assert evidence.usable and evidence.visible_ankles==2 and evidence.contact_world_y==0
    target=SupportPlaneEvidence(True,.7,0,.95,2,("test",)); expected=expected_support_pitch_deg(4,1.5,target); assert np.isclose(expected,np.degrees(np.arctan2(1.5,4)))
    _,coh=_camera_pose_from_params(4,1.5,0,expected,0); _,off=_camera_pose_from_params(4,1.5,0,0,0); intr=CameraIntrinsics.from_focal_mm(50,900,800)
    assert candidate_support_plane_score(PoseCandidate(intr,coh,4,1.5,50,.8,{}),target) > candidate_support_plane_score(PoseCandidate(intr,off,4,1.5,50,.8,{}),target)


def test_image_refinement_and_anchor():
    k=_kps(); assert np.allclose(subject_anchor(k),[500,520]); intr=CameraIntrinsics.from_focal_mm(50,1000,800); _,extr=_camera_pose_from_params(4,1.3,0,12,0); c=PoseCandidate(intr,extr,4,1.3,50,.7,{})
    refined=refine_camera_candidate(c,k,1000,800); assert np.isfinite(refined.extrinsics.position).all() and abs(refined.extrinsics.pitch-12)<=9
    class LM:
        def __init__(self,x,y,v): self.x=x; self.y=y; self.visibility=v
    lms=[LM(x,y,.9) for x,y,_ in k]; q=assess_landmarks(lms); assert q.visible_count==17 and q.anchor_confidence>.8 and semantic_anchor_pixels(lms,1000,1000)==(500,520)


def test_cue_modes_history_and_voice():
    def cue(text,reason=""): return PhotographerCue(CuePriority.PRIMARY,text,reason,"pose")
    cues=[cue("重心放到一条腿上。","释放对称感"),cue("手肘别夹死。","打开轮廓")]
    assert format_cues(cues,CueMode.CONCISE)==["重心放到一条腿上。"] and format_cues(cues,CueMode.NORMAL)==[c.cue for c in cues]
    assert "原因：释放对称感" in format_cues(cues,CueMode.TECHNICAL)[0] and primary_cue(cues)==cues[0].cue
    h=CueHistory(capacity=3); h.push([cue("第一步")]); h.push([cue("第二步")]); assert h.undo().summary=="第一步" and h.redo().summary=="第二步"
    text=voice_ready_text("重心放到一条腿上。   看我。"); assert text=="重心放到一条腿上。\n看我。" and ssml(text).startswith("<speak>")


def test_photographer_cues_are_geometry_driven():
    cues=generate_photographer_cues(_action(),_orientation(),_camera(),_composition()); assert cues and cues[0].reason and "重心" in speakable_summary(cues)
    left=generate_photographer_cues(_action(),_orientation(),_camera(),_composition(x=.2)); right=generate_photographer_cues(_action(),_orientation(),_camera(),_composition(x=.8)); assert any("右边" in c.cue for c in left) and any("左边" in c.cue for c in right)
    tight=generate_photographer_cues(_action(),_orientation(),_camera(.8),_composition()); assert any(c.category=="camera" for c in tight)


def test_pose_guidance_is_goal_oriented():
    guidance=_generate_pose_guidance(_action(),_orientation(),_composition()); titles=[s.title for s in guidance]; assert "先释放对称站姿" in titles and "把手臂从躯干上分开" in titles
    sitting=_action(knee_angle_avg=98,knee_angle_diff=6); assert any(s.title=="让双腿产生前后层次" for s in _generate_pose_guidance(sitting,_orientation(),_composition()))
    result=generate_suggestions(_action(),_orientation(),_camera(),_composition()); assert result.next_actions==["调整重心","打开身体轮廓","改变头部方向"]
    result2=generate_suggestions(_action(),_orientation(),_camera(),_composition(x=.75)); assert any("朝画面中央打开" in s.title for s in result2.suggestions) and "让动作朝负空间展开" in result2.creative_direction


def test_reference_reconstruction_contract():
    reference=_reference_pose(); current=_reference_pose(offset_x=20,offset_y=5)
    assert choose_semantic_anchor(reference).name=="hip_center"
    comp=build_reference_composition(reference,200,180); names={a.name for a in comp.anchors}; assert {"bbox_tl","bbox_center","bbox_br","hip_center"}.issubset(names) and comp.subject_scale>0
    deltas=compare_pose_to_reference(reference,current,200,180); nose=next(d for d in deltas if d.landmark=="nose"); assert nose.dx<0 and nose.dy<0
    cur=build_reference_composition(_reference_pose(offset_x=10,scale=.8),200,180); delta=composition_delta(comp,cur); assert delta["center_dx"]<0 and 0<delta["scale_ratio"]<1


def test_scene_rotation_solver_contract():
    from reverse_engineering.rotation_solver import estimate_rotation_candidates
    width,height,focal=2400,1600,50; intr=CameraIntrinsics.from_focal_mm(focal,width,height); _,extr=_camera_pose_from_params(5,1.4,12,-5,6); R=cv2.Rodrigues(extr.rvec)[0]
    def vp(axis):
        d=R@axis
        if d[2]<0:d=-d
        return float(intr.fx*d[0]/d[2]+intr.cx), float(intr.cy-intr.fy*d[1]/d[2])
    vps=tuple(VanishingPoint(*vp(a),cluster=i,support=12-i,confidence=.9,mean_line_residual_px=1) for i,a in enumerate((np.array([1.,0,0]),np.array([0,0,1.]),np.array([0,1.,0.]))))
    evidence=SceneGeometryEvidence(width,height,tuple(),((0,1),(2,3),(4,5)),vps,2,(0,1),6,.9)
    candidates=estimate_rotation_candidates(evidence,width,height,max_candidates=8); best=max(candidates,key=lambda c:c.scene_score); assert candidates and abs(best.focal_length_mm-focal)<2 and abs(best.extrinsics.yaw-12)<1.5 and abs(best.extrinsics.pitch+5)<1.5 and abs(best.extrinsics.roll-6)<1.5


def test_exif_orientation_normalizes_landscape_storage_to_portrait_display(tmp_path):
    raw = cv2.imencode('.jpg', np.full((60, 40, 3), 220, dtype=np.uint8))[1].tobytes()
    tiff = bytearray(b'II*\x00\x08\x00\x00\x00\x01\x00')
    tiff += b'\x12\x01\x03\x00\x01\x00\x00\x00\x06\x00\x00\x00' + b'\x00\x00\x00\x00'
    exif = b'Exif\x00\x00' + bytes(tiff)
    segment = b'\xff\xe1' + (len(exif) + 2).to_bytes(2, 'big') + exif
    path = tmp_path / 'portrait-exif.jpg'
    path.write_bytes(raw[:2] + segment + raw[2:])
    image = load_image(path)
    assert image is not None and image.shape[:2] == (40, 60)
    assert frame_orientation(image) == 'portrait'
    probe = np.zeros((2, 3, 3), dtype=np.uint8); assert _apply_exif_orientation(probe, 6).shape[:2] == (3, 2)


def test_field_mode_styles_scope_foreground_and_primary_surface():
    from gui.field_mode import FieldModeWidget
    theme = FieldModeWidget._THEME
    assert theme['bg'] == '#10151c'
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    widget = FieldModeWidget(); css = widget.styleSheet()
    assert 'QWidget#fieldMode QLabel' in css and 'QLabel#primaryCue' in css and 'QComboBox QAbstractItemView' in css
    widget.close()


def test_v3_phase2_target_plan_distinguishes_framing_and_pose():
    reference = build_reference_composition(_reference_pose(), 200, 180)
    current_pose = _reference_pose(offset_x=-18, offset_y=10, scale=.8)
    current = build_reference_composition(current_pose, 200, 180)
    deltas = compare_pose_to_reference(_reference_pose(), current_pose, 200, 180)
    plan = build_reference_target_plan(reference, current, deltas)
    text = plan.as_text()
    assert plan.framing_actions and plan.pose_actions
    assert '构图：' in text and '姿态：' in text


def test_v3_phase2_canvas_target_geometry_is_resolution_independent():
    reference = _reference_pose(offset_x=10, offset_y=5, scale=1.0)
    current = _reference_pose(offset_x=-12, offset_y=9, scale=.82)
    reference_comp = build_reference_composition(reference, 200, 180)
    current_comp = build_reference_composition(current, 800, 1200)
    deltas = compare_pose_to_reference(reference, current, 800, 1200)
    assert deltas
    assert all(0.0 <= d.target_x <= 1.0 and 0.0 <= d.target_y <= 1.0 for d in deltas)
    nose = next(d for d in deltas if d.landmark == 'nose')
    assert np.isclose(nose.target_x, reference.landmarks[0].x / reference.image_width)
    assert np.isclose(nose.target_y, reference.landmarks[0].y / reference.image_height)


def test_v3_phase2_canvas_exposes_reference_target_api():
    from PySide6.QtWidgets import QApplication
    from gui.canvas import ImageCanvas
    app = QApplication.instance() or QApplication([])
    canvas = ImageCanvas()
    reference = build_reference_composition(_reference_pose(), 200, 180)
    current = build_reference_composition(_reference_pose(offset_x=12), 200, 180)
    deltas = compare_pose_to_reference(_reference_pose(), _reference_pose(offset_x=12), 200, 180)
    canvas.set_reference_target(reference, current, deltas, visible=True)
    assert canvas._reference_target is reference
    assert canvas._reference_current is current
    assert canvas._show_reference_target is True
    canvas.clear_reference_target()
    assert canvas._reference_target is None and not canvas._show_reference_target
    canvas.close()
