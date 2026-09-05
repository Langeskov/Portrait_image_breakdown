"""Photography reverse-engineering pipeline."""
from __future__ import annotations

import math
import cv2
import numpy as np
from reverse_engineering.data_types import ReverseEngineeringResult, CompositionResult, EstimatedValue, CameraAction, CameraPoseResult
from reverse_engineering.perspective import analyze_perspective
from reverse_engineering.camera_pose import estimate_camera_pose
from reverse_engineering.focal_length import estimate_focal_length
from reverse_engineering.depth_of_field import analyze_depth_of_field
from reverse_engineering.motion_blur import analyze_motion_blur
from reverse_engineering.shooting_technique import classify_techniques
from reverse_engineering.simulation import optimize_parameters
from reverse_engineering.depth_provider import MonocularDepthProvider
from core.pose_detector import PoseResult, LandmarkIndex as LI


def _extract_keypoints_pixels(pose: PoseResult) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.visibility] for lm in pose.landmarks[:17]], dtype=float)


def _analyze_composition_extended(image, pose=None, bbox=None):
    h, w = image.shape[:2]
    if pose is not None:
        visible = [(lm.world_x, lm.world_y) for lm in pose.landmarks[:17] if lm.visibility > .4]
        sx = float(np.mean([p[0] for p in visible])) if visible else .5
        sy = float(np.mean([p[1] for p in visible])) if visible else .5
    else:
        sx, sy = .5, .5
    if bbox:
        x1,y1,x2,y2=bbox; subject_scale=max(0.,min(1.,(x2-x1)*(y2-y1)/max(w*h,1)))
    elif pose:
        vp=[(lm.world_x,lm.world_y) for lm in pose.landmarks[:17] if lm.visibility>.4]
        subject_scale=max(0.,min(1.,(max(p[0] for p in vp)-min(p[0] for p in vp))*(max(p[1] for p in vp)-min(p[1] for p in vp)))) if len(vp)>=4 else .3
    else:
        subject_scale=.3
    headroom=pose.landmarks[LI.NOSE].world_y if pose and pose.is_visible(LI.NOSE) else sy
    look_room="right" if sx<.4 else "left" if sx>.6 else "balanced"
    neg_space=1-sx if look_room=="right" else sx if look_room=="left" else .5
    styles=[]; min_dist=min(abs(sx-tx)+abs(sy-ty) for tx in (1/3,2/3) for ty in (1/3,2/3)); ts=max(0.,1-min_dist/.35)
    if ts>.4: styles.append({"name":"rule_of_thirds","confidence":round(ts,2)})
    cd=math.hypot(sx-.5,sy-.5)
    if cd<.12: styles.append({"name":"center","confidence":round(1-cd/.12,2)})
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); small=cv2.resize(gray,(64,64)).astype(float)
    sym=1-np.abs(small-np.fliplr(small)).mean()/(small.max()-small.min()+1)
    if sym>.7: styles.append({"name":"symmetry","confidence":round(float(sym),2)})
    if neg_space>.6: styles.append({"name":"negative_space","confidence":round(neg_space*.8,2)})
    if subject_scale<.4: styles.append({"name":"depth_layering","confidence":.5})
    styles.sort(key=lambda s:s["confidence"],reverse=True)
    return CompositionResult(styles=styles[:5],subject_position=(round(sx,3),round(sy,3)),subject_scale=round(subject_scale,4),headroom=round(headroom,3),look_room=look_room,negative_space_ratio=round(neg_space,3))


def _generate_camera_actions(result, sim_candidates=None):
    actions=[]; sx,_=result.composition.subject_position; min_dist=min(abs(sx-tx) for tx in (1/3,2/3))
    if min_dist>.15:
        actions.append(CameraAction("MOVE_RIGHT" if sx<.5 else "MOVE_LEFT",["subject not near thirds verticals",f"x={sx:.2f}"],"adjust horizontal framing",priority=2))
    if result.composition.subject_scale>.7:
        actions.append(CameraAction("MOVE_BACKWARD",[f"subject {result.composition.subject_scale:.0%} of frame"],"reduce subject scale",priority=2))
    elif result.composition.subject_scale<.1:
        actions.append(CameraAction("MOVE_FORWARD",[f"subject only {result.composition.subject_scale:.0%}"],"increase subject presence",priority=2))
    if sim_candidates and len(sim_candidates)>1:
        alt=sim_candidates[1]; actions.append(CameraAction("CHANGE_FOCAL_LENGTH",[f"alternative {alt.focal_equiv_35mm:.1f}mm at {alt.distance:.2f}m"],"explore another valid focal/distance solution",priority=1))
    actions += [CameraAction("WAIT",["observe subject movement"],"wait for decisive moment",priority=0),CameraAction("CAPTURE",["current solution is acceptable"],"take the shot",priority=0)]
    actions.sort(key=lambda a:a.priority,reverse=True); return actions[:8]


def _camera_pose_from_candidate(candidate) -> CameraPoseResult:
    c=candidate
    return CameraPoseResult(
        camera_height=EstimatedValue(round(c.height,2),unit="m",range_min=max(.25,round(c.height-.25,2)),range_max=min(2.5,round(c.height+.25,2)),confidence=min(.75,c.score),basis=["bounded 2D-pose camera fit"]),
        camera_distance=EstimatedValue(round(c.distance,2),unit="m",range_min=max(.5,round(c.distance*.75,2)),range_max=round(c.distance*1.25,2),confidence=min(.75,c.score),basis=["bounded 2D-pose camera fit","focal/distance ambiguity retained"]),
        camera_pitch=EstimatedValue(round(c.extrinsics.pitch,1),unit="deg",range_min=round(c.extrinsics.pitch-4,1),range_max=round(c.extrinsics.pitch+4,1),confidence=min(.7,c.score),basis=["joint reprojection fit"]),
        camera_yaw=EstimatedValue(round(c.extrinsics.yaw,1),unit="deg",range_min=round(c.extrinsics.yaw-4,1),range_max=round(c.extrinsics.yaw+4,1),confidence=min(.7,c.score),basis=["joint reprojection fit"]),
        camera_roll=EstimatedValue(round(c.extrinsics.roll,1),unit="deg",range_min=round(c.extrinsics.roll-4,1),range_max=round(c.extrinsics.roll+4,1),confidence=min(.5,c.score),basis=["joint reprojection fit; image roll estimator remains separate"]),
    )


class ReverseEngineeringEngine:
    def __init__(self, enable_simulation=True):
        self._enable_simulation=enable_simulation; self._depth_provider=MonocularDepthProvider()

    def analyze(self,image,pose=None,bbox=None):
        uncertainties=[]; h,w=image.shape[:2]
        composition=_analyze_composition_extended(image,pose,bbox); perspective=analyze_perspective(image)
        fitted=[]
        if pose:
            kp=_extract_keypoints_pixels(pose)
            fitted=optimize_parameters(w,h,composition.subject_scale,composition.subject_position,perspective.perspective_strength.value,kp,num_candidates=5,subject_bbox=bbox)
        if fitted:
            camera_pose=_camera_pose_from_candidate(fitted[0])
            focal_length=estimate_focal_length(perspective.perspective_strength.value,perspective.perspective_type.value,composition.subject_scale,fitted)
        else:
            camera_pose=estimate_camera_pose(pose,perspective.vanishing_points,image=image,subject_bbox=bbox) if pose else CameraPoseResult(
                camera_height=EstimatedValue(1.5,unit="m",confidence=.1,basis=["no pose"]),camera_distance=EstimatedValue(4.,unit="m",confidence=.1,basis=["no pose"]),camera_pitch=EstimatedValue(0.,unit="deg",confidence=.1,basis=["no pose"]),camera_yaw=EstimatedValue(0.,unit="deg",confidence=.1,basis=["no pose"]),camera_roll=EstimatedValue(0.,unit="deg",confidence=.1,basis=["no pose"]))
            focal_length=estimate_focal_length(perspective.perspective_strength.value,perspective.perspective_type.value,composition.subject_scale)
        depth_of_field=analyze_depth_of_field(image,pose,bbox); motion_blur=analyze_motion_blur(image,bbox)
        shooting_techniques=classify_techniques(perspective,camera_pose,focal_length,depth_of_field,motion_blur,composition,composition.subject_scale)
        conf_vals=[focal_length.category.confidence,focal_length.equivalent_35mm.confidence,depth_of_field.dof_type.confidence,motion_blur.blur_type.confidence,camera_pose.camera_height.confidence,camera_pose.camera_distance.confidence]
        overall_conf=float(np.mean(conf_vals))
        uncertainties += ["exact focal length cannot be uniquely determined from a single image","sensor format and crop status may be unknown","aperture is inferred only from blur/depth characteristics","camera height/distance remain coupled without scene scale or depth"]
        result=ReverseEngineeringResult(image_size=(w,h),subject_bbox=bbox,subject_keypoints=pose.landmarks[:17] if pose else None,subject_scale=composition.subject_scale,edge_lines=perspective.line_segments,blur_regions={},perspective=perspective,camera_pose=camera_pose,focal_length=focal_length,depth_of_field=depth_of_field,motion_blur=motion_blur,composition=composition,shooting_techniques=shooting_techniques,overall_confidence=overall_conf,uncertainties=uncertainties,_sim_candidates=fitted)
        result._camera_actions=_generate_camera_actions(result,fitted); return result
