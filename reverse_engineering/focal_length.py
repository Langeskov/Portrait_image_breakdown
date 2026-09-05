"""Focal-length estimation backed by the camera candidate family."""
from __future__ import annotations
from reverse_engineering.data_types import EstimatedValue, FocalLengthResult
from reverse_engineering.geometry import PoseCandidate


def _category(focal_mm: float) -> str:
    if focal_mm < 35: return "wide"
    if focal_mm < 60: return "normal"
    if focal_mm < 105: return "short_telephoto"
    return "telephoto"


def estimate_focal_length(perspective_strength: float, perspective_type: str, subject_scale: float, candidates: list[PoseCandidate] | None = None) -> FocalLengthResult:
    """Use the best fitted camera as the displayed focal length; retain family spread as uncertainty."""
    if candidates:
        ordered=sorted(candidates,key=lambda c:(-float(c.score),float(c.losses.get("mean_reprojection_px",1e9))))
        best=ordered[0]; top=ordered[:min(5,len(ordered))]
        vals=[float(c.focal_equiv_35mm) for c in top]
        scores={}
        for c in top:
            cat=_category(c.focal_equiv_35mm); scores[cat]=scores.get(cat,0)+max(float(c.score),.01)
        cat=_category(best.focal_equiv_35mm); total=sum(scores.values()) or 1
        cat_conf=min(.8,max(.1,scores.get(cat,0)/total)); spread=max(vals)-min(vals)
        fit_conf=min(.8,max(.2,float(best.score))); mm_conf=min(.75,fit_conf if spread<=15 else fit_conf*15/spread)
        return FocalLengthResult(category=EstimatedValue(cat,confidence=cat_conf,basis=["best bounded 2D-pose camera fit",f"best score={best.score:.2f}"]),equivalent_35mm=EstimatedValue(round(best.focal_equiv_35mm,1),unit="mm",range_min=round(min(vals),1),range_max=round(max(vals),1),confidence=mm_conf,basis=["best geometric candidate","candidate spread retained as uncertainty","single-image focal length remains under-constrained"]))
    scores={"wide":0.,"normal":0.,"short_telephoto":0.,"telephoto":0.}
    if perspective_type=="wide" or perspective_strength>.55: scores["wide"]+=.35
    elif perspective_type=="telephoto" or perspective_strength<.3: scores["short_telephoto"]+=.25; scores["telephoto"]+=.15
    else: scores["normal"]+=.25; scores["short_telephoto"]+=.20
    if subject_scale>.55: scores["short_telephoto"]+=.20
    elif subject_scale<.15: scores["wide"]+=.15
    else: scores["normal"]+=.10
    cat=max(scores,key=scores.get); fmap={"wide":(28,18,35),"normal":(50,35,70),"short_telephoto":(85,70,105),"telephoto":(135,105,300)}
    est,lo,hi=fmap[cat]; conf=min(.55,scores[cat]+.1)
    return FocalLengthResult(category=EstimatedValue(cat,confidence=conf,basis=["heuristic fallback","no pose candidate family"]),equivalent_35mm=EstimatedValue(est,unit="mm",range_min=lo,range_max=hi,confidence=conf*.6,basis=["heuristic fallback","not uniquely observable from one image"]))
