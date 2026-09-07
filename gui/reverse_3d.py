"""3D reverse-engineering workspace with 3D->2D validation."""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPolygonF, QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QLabel, QDoubleSpinBox,
    QGroupBox, QFormLayout, QListWidget, QListWidgetItem, QSizePolicy,
    QPushButton,
)

from reverse_engineering.scene import SceneModel, SceneCamera
from reverse_engineering.data_types import ReverseEngineeringResult
from reverse_engineering.projection import project_subject, ProjectionPreviewResult
from core.pose_detector import PoseResult

BG = QColor("#F8FAFC")
GRID = QColor("#CBD5E1")
GRID_MAJOR = QColor("#94A3B8")
AXIS = QColor("#64748B")
TEXT = QColor("#334155")
CAMERA = QColor("#2563EB")
CAMERA_BODY = QColor("#0F172A")
LENS = QColor("#475569")
FRUSTUM = QColor(37, 99, 235, 145)
FRUSTUM_SOFT = QColor(37, 99, 235, 48)
SUBJECT = QColor("#16A34A")
SELECTED = QColor("#D97706")
ALT = QColor(148, 163, 184, 150)
OBSERVED = QColor("#D97706")
SECONDARY = QColor("#7C3AED")


class SceneView(QWidget):
    camera_changed = Signal()

    def __init__(self, scene: Optional[SceneModel] = None, parent=None):
        super().__init__(parent)
        self.scene = scene or SceneModel()
        self.setMinimumSize(580, 440)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._azimuth = 34.0
        self._elevation = 25.0
        self._distance_scale = 96.0
        self._last_pos = None

    def set_scene(self, scene: SceneModel):
        self.scene = scene
        self.update()

    def reset_view(self):
        self._azimuth, self._elevation, self._distance_scale = 34.0, 25.0, 96.0
        self.update()

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self._last_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._last_pos is None:
            return
        delta = event.position() - self._last_pos
        self._last_pos = event.position()
        self._azimuth += float(delta.x()) * 0.45
        self._elevation = max(-80.0, min(80.0, self._elevation - float(delta.y()) * 0.35))
        self.update()

    def mouseReleaseEvent(self, event):
        self._last_pos = None
        self.unsetCursor()

    def wheelEvent(self, event):
        factor = 1.12 if event.angleDelta().y() > 0 else 0.89
        self._distance_scale = max(34.0, min(200.0, self._distance_scale * factor))
        self.update()

    def _view_basis(self):
        az, el = math.radians(self._azimuth), math.radians(self._elevation)
        forward = np.array([math.cos(el) * math.cos(az), math.sin(el), math.sin(az) * math.cos(el)])
        forward /= max(np.linalg.norm(forward), 1e-9)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(right, forward)
        up /= max(np.linalg.norm(up), 1e-9)
        return right, up, forward

    def _project(self, point):
        right, up, forward = self._view_basis()
        q = np.asarray(point, dtype=float) - self.scene.camera_target()
        x, y, z = float(np.dot(q, right)), float(np.dot(q, up)), float(np.dot(q, forward))
        scale = self._distance_scale / max(6.0, 8.0 + z * 0.08)
        return QPointF(self.width() * 0.5 + x * scale, self.height() * 0.54 - y * scale)

    def _line(self, painter, a, b, color, width=1, dash=False):
        painter.setPen(QPen(color, width, Qt.DashLine if dash else Qt.SolidLine))
        painter.drawLine(self._project(a), self._project(b))

    def _camera_axes(self):
        forward = self.scene.camera.forward()
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= max(np.linalg.norm(up), 1e-9)
        roll = math.radians(self.scene.camera.roll)
        c, s = math.cos(roll), math.sin(roll)
        return forward, right * c + up * s, -right * s + up * c

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), BG)
        self._draw_grid(painter)
        self._draw_subject(painter)
        self._draw_candidate_cameras(painter)
        self._draw_camera(painter)
        self._draw_labels(painter)
        painter.end()

    def _draw_grid(self, painter):
        size, spacing = max(24.0, float(self.scene.ground_size)), 1.0
        lines = int(size / spacing)
        for i in range(-lines, lines + 1):
            major = i % 5 == 0
            color, width = (GRID_MAJOR, 1.7) if major else (GRID, 1)
            x, z = i * spacing, i * spacing
            self._line(painter, (x, 0, -size), (x, 0, size), color, width)
            self._line(painter, (-size, 0, z), (size, 0, z), color, width)
        self._line(painter, (-size, 0, 0), (size, 0, 0), AXIS, 2)
        self._line(painter, (0, 0, -size), (0, 0, size), AXIS, 2)
        self._line(painter, (0, 0, 0), (0, 2.5, 0), AXIS, 2)

    def _draw_subject(self, painter):
        pts = self.scene.subject.proxy_points() + np.array([
            self.scene.subject.center_x, self.scene.subject.center_y, self.scene.subject.center_z
        ])
        links = [(0,1),(0,2),(1,3),(2,4),(2,3),(3,5),(4,6),(5,6),(6,7)]
        painter.setPen(QPen(SUBJECT, 3))
        for a, b in links:
            painter.drawLine(self._project(pts[a]), self._project(pts[b]))
        painter.setBrush(QBrush(SUBJECT))
        for p in pts:
            painter.drawEllipse(self._project(p), 4, 4)

    def _draw_camera_marker(self, painter, position, color, radius=5):
        p = self._project(position)
        painter.setPen(QPen(color, 2))
        painter.setBrush(QBrush(BG))
        painter.drawEllipse(p, radius, radius)
        painter.drawLine(p.x() - 8, p.y(), p.x() + 8, p.y())
        painter.drawLine(p.x(), p.y() - 8, p.x(), p.y() + 8)

    def _draw_camera_body(self, painter, position):
        pos = np.asarray(position, dtype=float)
        forward, right, up = self._camera_axes()
        body_len, body_w, body_h = 0.58, 0.36, 0.25
        back, front = pos - forward * body_len * 0.5, pos + forward * body_len * 0.5
        corners = [
            back + right*body_w + up*body_h, back - right*body_w + up*body_h,
            back - right*body_w - up*body_h, back + right*body_w - up*body_h,
            front + right*body_w + up*body_h, front - right*body_w + up*body_h,
            front - right*body_w - up*body_h, front + right*body_w - up*body_h,
        ]
        painter.setPen(QPen(CAMERA_BODY, 2))
        painter.setBrush(QBrush(QColor(255,255,255,245)))
        for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            painter.drawLine(self._project(corners[a]), self._project(corners[b]))
        base, tip, r = front + forward*0.03, front + forward*0.32, 0.16
        rings = []
        for c in (base, tip):
            rings.append([c+right*r+up*r, c-right*r+up*r, c-right*r-up*r, c+right*r-up*r])
        painter.setPen(QPen(LENS, 2))
        for ring in rings:
            for i in range(4):
                painter.drawLine(self._project(ring[i]), self._project(ring[(i+1)%4]))
        for i in range(4):
            painter.drawLine(self._project(rings[0][i]), self._project(rings[1][i]))

    def _draw_candidate_cameras(self, painter):
        target = self.scene.camera_target()
        for i, c in enumerate(self.scene.candidate_solutions):
            if i == self.scene.selected_candidate:
                continue
            yaw = math.radians(float(getattr(c.extrinsics, "yaw", 0.0)))
            p = np.array([
                self.scene.subject.center_x + math.sin(yaw) * c.distance,
                c.height,
                self.scene.subject.center_z - math.cos(yaw) * c.distance,
            ])
            self._line(painter, p, target, ALT, 1, True)
            self._draw_camera_marker(painter, p, ALT, 4)

    def _draw_camera(self, painter):
        position = self.scene.camera_position()
        target = self.scene.camera_target()
        forward, right, up = self._camera_axes()
        self._draw_camera_body(painter, position)
        subject_depth = float(np.dot(target - position, forward))
        depth = max(2.5, subject_depth if subject_depth > 0 else self.scene.camera.distance)
        depth = min(depth * 1.06, 60.0)
        near = 0.55
        far = position + forward * depth
        fov_x = math.radians(self.scene.camera.horizontal_fov_deg)
        fov_y = math.radians(self.scene.camera.vertical_fov_deg)
        hw, hh = math.tan(fov_x * 0.5) * depth, math.tan(fov_y * 0.5) * depth
        frame = [far+right*hw+up*hh, far-right*hw+up*hh, far-right*hw-up*hh, far+right*hw-up*hh]
        self._line(painter, position, position+forward*near, CAMERA, 3)
        self._line(painter, position+forward*near, far, CAMERA, 2, True)
        self._line(painter, position, target, CAMERA, 2)
        painter.setPen(QPen(FRUSTUM, 2))
        painter.setBrush(QBrush(FRUSTUM_SOFT))
        painter.drawPolygon(QPolygonF([self._project(p) for p in frame]))
        for p in frame: self._line(painter, position+forward*near, p, FRUSTUM, 2)
        for i in range(4): self._line(painter, frame[i], frame[(i+1)%4], FRUSTUM, 2)
        self._line(painter, frame[0], frame[2], FRUSTUM_SOFT, 1, True)
        self._line(painter, frame[1], frame[3], FRUSTUM_SOFT, 1, True)
        self._line(painter, position+forward*near, far, FRUSTUM, 1)
        tp = self._project(target)
        painter.setPen(QPen(SELECTED, 2)); painter.setBrush(QBrush(QColor(255,255,255,210)))
        painter.drawEllipse(tp, 5, 5); painter.drawLine(tp.x()-11,tp.y(),tp.x()+11,tp.y()); painter.drawLine(tp.x(),tp.y()-11,tp.x(),tp.y()+11)
        self._line(painter, position, position + up*1.0, SELECTED if abs(self.scene.camera.roll)>1 else AXIS, 2)

    def _draw_labels(self, painter):
        painter.setPen(TEXT); painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(12,22,"3D scene — drag to orbit, wheel to zoom")
        painter.drawText(12,40,"Camera / frustum / optical axis / subject target")
        c=self.scene.camera
        painter.drawText(12,58,f"FOV {c.horizontal_fov_deg:.1f}° × {c.vertical_fov_deg:.1f}°   YAW {c.yaw:+.1f}°   PITCH {c.pitch:+.1f}°   ROLL {c.roll:+.1f}°")


class ProjectionPreview(QWidget):
    """Source image plus observed multi-person poses and predicted primary framing."""
    def __init__(self, parent=None):
        super().__init__(parent); self.setMinimumHeight(300); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._pixmap: Optional[QPixmap] = None; self._projection: Optional[ProjectionPreviewResult] = None
        self._observed_bbox = None; self._observed_points = []; self._metrics = "No projection yet"

    def set_image(self, image: Optional[np.ndarray]):
        if image is None: self._pixmap = None
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
            self._pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy())
        self.update()

    def update_projection(self, scene, observed_bbox=None, observed_points=None):
        self._observed_bbox, self._observed_points = observed_bbox, observed_points or []
        if self._pixmap is None:
            self._projection = None; self._metrics = "No source image"; self.update(); return
        w, h = self._pixmap.width(), self._pixmap.height(); self._projection = project_subject(scene, w, h)
        if self._projection.bbox is None:
            self._metrics = "3D proxy is outside the camera-facing half-space"; self.update(); return
        x0,y0,x1,y1 = self._projection.bbox; metrics=[f"3D primary {max(0,x1-x0)/max(w,1):.0%} W × {max(0,y1-y0)/max(h,1):.0%} H"]
        if observed_bbox is not None and len(observed_bbox)>=4:
            ox0,oy0,ox1,oy1=map(float, observed_bbox[:4]); inter=max(0,min(x1,ox1)-max(x0,ox0))*max(0,min(y1,oy1)-max(y0,oy0))
            ap=max(0,x1-x0)*max(0,y1-y0); ao=max(0,ox1-ox0)*max(0,oy1-oy0); union=ap+ao-inter; iou=inter/union if union>1e-9 else 0
            dc=math.hypot((x0+x1-ox0-ox1)/2,(y0+y1-oy0-oy1)/2)/max(math.hypot(w,h),1); metrics += [f"bbox IoU {iou:.0%}", f"center Δ {dc:.1%}"]
        if len(self._observed_points)>1: metrics.append(f"{len(self._observed_points)} people detected")
        self._metrics=" · ".join(metrics); self.update()

    def _map(self,x,y,ox,oy,sx,sy): return QPointF(ox+float(x)*sx,oy+float(y)*sy)

    def paintEvent(self,event):
        painter=QPainter(self); painter.setRenderHint(QPainter.Antialiasing); painter.fillRect(self.rect(),QColor("#111827"))
        if self._pixmap is None:
            painter.setPen(QColor("#CBD5E1")); painter.drawText(self.rect(),Qt.AlignCenter,"Projection preview"); painter.end(); return
        area=QRectF(6,6,self.width()-12,self.height()-38); scaled=self._pixmap.scaled(area.size().toSize(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
        ox=area.x()+(area.width()-scaled.width())*0.5; oy=area.y()+(area.height()-scaled.height())*0.5; painter.drawPixmap(int(ox),int(oy),scaled)
        sx,sy=scaled.width()/max(self._pixmap.width(),1),scaled.height()/max(self._pixmap.height(),1)
        def rect_of(r): x0,y0,x1,y1=map(float,r[:4]); return QRectF(ox+x0*sx,oy+y0*sy,(x1-x0)*sx,(y1-y0)*sy)
        def pt(x,y): return self._map(x,y,ox,oy,sx,sy)
        for idx, points in enumerate(self._observed_points):
            points=np.asarray(points,dtype=float); links=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4)]; col=OBSERVED if idx==0 else SECONDARY
            painter.setPen(QPen(col,1.6))
            for a,b in links:
                if a<len(points) and b<len(points) and np.isfinite(points[[a,b]]).all(): painter.drawLine(pt(points[a,0],points[a,1]),pt(points[b,0],points[b,1]))
            painter.setBrush(QBrush(col))
            for p in points:
                if np.isfinite(p[:2]).all(): painter.drawEllipse(pt(p[0],p[1]),2.6,2.6)
        if self._observed_bbox is not None and len(self._observed_bbox)>=4:
            r=rect_of(self._observed_bbox); painter.setPen(QPen(OBSERVED,2,Qt.DashLine)); painter.setBrush(Qt.NoBrush); painter.drawRect(r); painter.setFont(QFont("Segoe UI",8)); painter.drawText(r.topLeft()+QPointF(3,-4),"OBSERVED PRIMARY")
        if self._projection and self._projection.bbox:
            r=rect_of(self._projection.bbox); painter.setPen(QPen(CAMERA,2)); painter.setBrush(Qt.NoBrush); painter.drawRect(r); painter.setFont(QFont("Segoe UI",8)); painter.drawText(r.topRight()+QPointF(-58,-4),"3D PRIMARY"); cx,cy=r.center().x(),r.center().y(); painter.drawLine(cx-6,cy,cx+6,cy); painter.drawLine(cx,cy-6,cx,cy+6)
        painter.setPen(QColor("#E5E7EB")); painter.setFont(QFont("Segoe UI",8)); painter.drawText(8,self.height()-10,self._metrics); painter.end()


class Reverse3DWorkspace(QWidget):
    camera_edited=Signal()

    def __init__(self,parent=None):
        super().__init__(parent); self.scene=SceneModel(); self._result:Optional[ReverseEngineeringResult]=None; self._source_image=None; self._observed_bbox=None; self._observed_points=[]
        root=QVBoxLayout(self); root.setContentsMargins(0,0,0,0); splitter=QSplitter(Qt.Horizontal)
        left=QWidget(); llo=QVBoxLayout(left); llo.setContentsMargins(0,0,0,0); self._view=SceneView(self.scene); llo.addWidget(self._view,1)
        reset=QPushButton("Reset 3D view"); reset.clicked.connect(self._view.reset_view); reset.setMaximumWidth(130); llo.addWidget(reset,0,Qt.AlignRight)
        splitter.addWidget(left); splitter.addWidget(self._build_panel()); splitter.setSizes([940,440]); splitter.setStretchFactor(0,1); splitter.setStretchFactor(1,0); root.addWidget(splitter)

    def _build_panel(self):
        panel=QWidget(); lo=QVBoxLayout(panel); lo.setContentsMargins(14,14,14,14); lo.setSpacing(10)
        title=QLabel("Camera reconstruction"); title.setFont(QFont("Segoe UI",15,QFont.Weight.Bold)); lo.addWidget(title)
        self._confidence=QLabel("No reverse-engineering result yet"); self._confidence.setStyleSheet("color:#64748B;"); lo.addWidget(self._confidence)
        group=QGroupBox("Camera"); form=QFormLayout(group)
        self._distance=self._spin(0.1,50,0.1," m"); self._height=self._spin(0.1,5,0.05," m"); self._yaw=self._spin(-180,180,1,"°"); self._pitch=self._spin(-90,90,1,"°"); self._roll=self._spin(-180,180,1,"°"); self._focal=self._spin(10,600,1," mm")
        for b in (self._distance,self._height,self._yaw,self._pitch,self._roll,self._focal): b.valueChanged.connect(self._camera_spin_changed)
        for label,b in (("Distance",self._distance),("Height",self._height),("Yaw",self._yaw),("Pitch",self._pitch),("Roll",self._roll),("Focal",self._focal)): form.addRow(label,b)
        lo.addWidget(group)
        hint=QLabel("提示：可以使用参数输入框的鼠标滚轮微调数值，快速复现画面。")
        hint.setWordWrap(True); hint.setStyleSheet("color:#64748B; font-size:9pt;"); lo.addWidget(hint)
        for box in (self._distance,self._height,self._yaw,self._pitch,self._roll,self._focal): box.setToolTip("可使用鼠标滚轮微调参数，实时观察 2D 画面复现效果。")
        pg=QGroupBox("2D projection preview"); plo=QVBoxLayout(pg); self._preview=ProjectionPreview(); plo.addWidget(self._preview)
        self._preview_metrics=QLabel("No projection yet"); self._preview_metrics.setStyleSheet("color:#475569; font-size:9pt;"); self._preview_metrics.setWordWrap(True); plo.addWidget(self._preview_metrics)
        note=QLabel("Orange = primary observed pose · purple = additional people · blue = current 3D projection"); note.setStyleSheet("color:#64748B; font-size:9pt;"); plo.addWidget(note); lo.addWidget(pg)
        cg=QGroupBox("Candidate solutions"); clo=QVBoxLayout(cg); self._candidates=QListWidget(); self._candidates.currentRowChanged.connect(self._select_candidate); clo.addWidget(self._candidates); lo.addWidget(cg,1)
        self._note=QLabel("Single-image reconstruction remains multi-solution. The 2D preview is a validation aid; only the primary person's 3D proxy is currently reconstructed."); self._note.setWordWrap(True); self._note.setStyleSheet("color:#64748B; font-size:9pt;"); lo.addWidget(self._note)
        return panel

    @staticmethod
    def _spin(lo,hi,step,suffix):
        box=QDoubleSpinBox(); box.setRange(lo,hi); box.setSingleStep(step); box.setDecimals(2 if step<0.1 else 1); box.setSuffix(suffix); return box

    def set_image(self,image:Optional[np.ndarray]):
        self._source_image=image; self._preview.set_image(image); self._resync_observed_geometry(); self._refresh_projection()

    def _resync_observed_geometry(self):
        pose=getattr(self._result,"subject_keypoints",None) if self._result else None
        if isinstance(pose, list): return

    def set_result(self,result:Optional[ReverseEngineeringResult],observed_bbox=None,observed_points=None):
        self._result=result; self._observed_bbox=observed_bbox; self._observed_points=observed_points or []; self.scene=SceneModel.from_reverse_result(result); self._view.set_scene(self.scene); self._sync_controls(); self._populate_candidates(); self._confidence.setText(f"Overall confidence: {result.overall_confidence:.0%}" if result else "No reverse-engineering result yet"); self._refresh_projection()

    def _sync_controls(self):
        c=self.scene.camera
        for box,value in ((self._distance,c.distance),(self._height,c.height),(self._yaw,c.yaw),(self._pitch,c.pitch),(self._roll,c.roll),(self._focal,c.focal_length_mm)):
            box.blockSignals(True); box.setValue(float(value)); box.blockSignals(False)

    def _populate_candidates(self):
        self._candidates.blockSignals(True); self._candidates.clear()
        for i,c in enumerate(self.scene.candidate_solutions): self._candidates.addItem(QListWidgetItem(f"#{i+1}  {c.focal_equiv_35mm} mm  /  {c.distance:.2f} m  /  h={c.height:.2f} m  /  score={c.score:.2f}"))
        if self.scene.candidate_solutions: self._candidates.setCurrentRow(self.scene.selected_candidate)
        self._candidates.blockSignals(False)

    def _select_candidate(self,row):
        if row<0 or not self.scene.candidate_solutions:return
        self.scene.set_candidate(row); self._sync_controls(); self._view.update(); self._refresh_projection(); self.camera_edited.emit()

    def _camera_spin_changed(self):
        self.scene.camera=SceneCamera(distance=float(self._distance.value()),height=float(self._height.value()),yaw=float(self._yaw.value()),pitch=float(self._pitch.value()),roll=float(self._roll.value()),focal_length_mm=float(self._focal.value())); self._view.set_scene(self.scene); self._refresh_projection(); self.camera_edited.emit()

    def _refresh_projection(self):
        self._preview.update_projection(self.scene,self._observed_bbox,self._observed_points); self._preview_metrics.setText(self._preview._metrics)

    def update_results(self,bundle):
        if bundle.reverse_result:
            pose=getattr(bundle,"pose",None)
            if pose is not None:
                people=getattr(pose,"persons",None) or [pose]
                if self._source_image is not None:
                    tw,th=self._source_image.shape[1],self._source_image.shape[0]; people=[p.rescaled(tw,th) if (p.image_width,p.image_height)!=(tw,th) else p for p in people]
                self._observed_points=[np.array([[lm.x,lm.y] for lm in p.landmarks[:17]],dtype=float) for p in people]; self._observed_bbox=getattr(people[0],"bbox",None)
            self.set_result(bundle.reverse_result,self._observed_bbox,self._observed_points)
        elif getattr(bundle,"pose",None):
            pose=bundle.pose; people=getattr(pose,"persons",None) or [pose]
            if self._source_image is not None:
                tw,th=self._source_image.shape[1],self._source_image.shape[0]; people=[p.rescaled(tw,th) if (p.image_width,p.image_height)!=(tw,th) else p for p in people]
            self._observed_points=[np.array([[lm.x,lm.y] for lm in p.landmarks[:17]],dtype=float) for p in people]; self._observed_bbox=getattr(people[0],"bbox",None); self._refresh_projection()
