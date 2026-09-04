"""MainWindow - Workspace architecture"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QKeySequence, QPalette, QColor, QAction
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QToolBar, QFileDialog, QLabel, QStatusBar, QMessageBox, QCheckBox, QComboBox,
    QApplication, QTabBar, QStackedWidget, QScrollArea)

THEME = dict(window="#F5F6F8", panel="#FFFFFF", border="#D9DDE3", text="#1F2937",
             text2="#6B7280", accent="#2563EB", toolbar="#FFFFFF")

def apply_light_theme(app):
    p = QPalette()
    for role, color in [(QPalette.Window, THEME["window"]), (QPalette.WindowText, THEME["text"]),
                        (QPalette.Base, THEME["panel"]), (QPalette.Text, THEME["text"]),
                        (QPalette.Button, THEME["toolbar"]), (QPalette.ButtonText, THEME["text"]),
                        (QPalette.Highlight, THEME["accent"]), (QPalette.HighlightedText, "#FFFFFF")]:
        p.setColor(role, QColor(color))
    app.setPalette(p)

class AnalysisWorker(QThread):
    finished = Signal(object)
    error = Signal(str)
    def __init__(self, det, eng, img):
        super().__init__(); self._d=det; self._e=eng; self._i=img
    def run(self):
        try:
            pose=self._d.detect(self._i)
            if pose is None: self.error.emit("No person detected"); return
            self.finished.emit((pose, self._e.analyze(self._i, pose, pose.bbox)))
        except Exception as e: self.error.emit(str(e))

class Workspace(QWidget):
    def update_results(self, pose, re): pass

from gui.canvas import ImageCanvas
from gui.panels import AnalysisPanel, SuggestionPanel

class Analysis2DWorkspace(Workspace):
    def __init__(self, parent=None):
        super().__init__(parent); lo=QHBoxLayout(self); lo.setContentsMargins(0,0,0,0)
        sp=QSplitter(Qt.Horizontal)
        self._ap=AnalysisPanel(); sp.addWidget(self._ap)
        self._cv=ImageCanvas(); sp.addWidget(self._cv)
        self._sp=SuggestionPanel(); sp.addWidget(self._sp)
        sp.setSizes([280,600,300]); sp.setStretchFactor(1,1); lo.addWidget(sp)
    def set_image(self, img): self._cv.set_image(img)
    def update_results(self, pose, re):
        self._cv.set_pose(pose)
        vis=sum(1 for lm in pose.landmarks[:17] if lm.visibility>0.4)
        self._ap.update_pose(pose.detection_confidence, vis)
    def set_overlay(self, **kw): self._cv.set_overlay_options(**kw)

class Reverse3DWorkspace(Workspace):
    def __init__(self, parent=None):
        super().__init__(parent); lo=QVBoxLayout(self); lo.setContentsMargins(16,16,16,16)
        lb=QLabel("3D Reverse Engineering Workspace")
        lb.setFont(QFont("Segoe UI",16,QFont.Weight.Bold))
        lb.setAlignment(Qt.AlignmentFlag.AlignCenter); lo.addWidget(lb)
        info=QLabel("Displays: Subject position, Camera frustum, Ground plane, Candidate solutions\nRequires: OpenGL 3D renderer (future phase)")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter); lo.addWidget(info)

class ResultsWorkspace(Workspace):
    def __init__(self, parent=None):
        super().__init__(parent); lo=QVBoxLayout(self); lo.setContentsMargins(16,16,16,16)
        self._rl=QLabel("No results yet."); self._rl.setFont(QFont("Consolas",10))
        self._rl.setAlignment(Qt.AlignmentFlag.AlignTop); self._rl.setWordWrap(True)
        sc=QScrollArea(); sc.setWidget(self._rl); sc.setWidgetResizable(True); lo.addWidget(sc)
    def update_results(self, pose, re): self._rl.setText(re.report())

from core.orientation import analyze_orientation
from core.action_classifier import classify_action

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Portrait Image Breakdown")
        self.setMinimumSize(1200,700); self.resize(1400,800)
        from core.pose_detector import PoseDetector
        from reverse_engineering.engine import ReverseEngineeringEngine
        self._det=PoseDetector(); self._eng=ReverseEngineeringEngine()
        self._img=None; self._wk=None
        tb=QToolBar("Main"); tb.setMovable(False); self.addToolBar(tb)
        ao=QAction("Open Image",self); ao.setShortcut(QKeySequence.Open)
        ao.triggered.connect(self._open); tb.addAction(ao)
        tb.addSeparator(); tb.addWidget(QLabel("  Dataset: "))
        self._cb=QComboBox(); self._cb.setMinimumWidth(200)
        self._cb.currentTextChanged.connect(self._sel); tb.addWidget(self._cb)
        tb.addSeparator()
        self._chk=QCheckBox("RE Overlay"); self._chk.stateChanged.connect(self._tog); tb.addWidget(self._chk)
        self._tabs=QTabBar()
        self._tabs.addTab("2D Analysis"); self._tabs.addTab("3D Reverse Engineering"); self._tabs.addTab("Results")
        self._tabs.currentChanged.connect(self._sw)
        self._ws=QStackedWidget()
        self._w2=Analysis2DWorkspace(); self._w3=Reverse3DWorkspace(); self._wr=ResultsWorkspace()
        self._ws.addWidget(self._w2); self._ws.addWidget(self._w3); self._ws.addWidget(self._wr)
        cen=QWidget(); ml=QVBoxLayout(cen); ml.setContentsMargins(0,0,0,0); ml.setSpacing(0)
        ml.addWidget(self._tabs); ml.addWidget(self._ws); self.setCentralWidget(cen)
        self._st=QStatusBar(); self.setStatusBar(self._st); self._st.showMessage("Ready")
        self._dd=Path(__file__).parent.parent/"dataset"
        if not self._dd.exists(): self._dd=Path.cwd()/"dataset"
        self._ld(); self.setAcceptDrops(True)
    def _ld(self):
        self._cb.blockSignals(True); self._cb.clear(); self._cb.addItem("Select image...")
        if self._dd.exists():
            for f in sorted(p.name for p in self._dd.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".webp")):
                self._cb.addItem(f)
        self._cb.blockSignals(False)
    def _sel(self, n):
        if n and n!="Select image...":
            p=self._dd/n
            if p.exists(): self._la(str(p))
    def _open(self):
        p,_=QFileDialog.getOpenFileName(self,"Select Image",str(self._dd),"Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        if p: self._la(p)
    def _la(self, path):
        img=cv2.imread(path)
        if img is None: QMessageBox.warning(self,"Error","Cannot read"); return
        self._img=img; self._w2.set_image(img)
        self._st.showMessage("Analyzing: "+os.path.basename(path)+" ...")
        if self._wk and self._wk.isRunning(): self._wk.terminate(); self._wk.wait()
        self._wk=AnalysisWorker(self._det,self._eng,img)
        self._wk.finished.connect(self._done); self._wk.error.connect(self._err); self._wk.start()
    def _done(self, r):
        pose,re=r
        for i in range(self._ws.count()):
            w=self._ws.widget(i)
            if hasattr(w,"update_results"): w.update_results(pose,re)
        o=analyze_orientation(pose); a=classify_action(pose)
        self._w2._ap.update_orientation(o); self._w2._ap.update_action(a)
        self._w2._ap.update_camera_result(re); self._w2._ap.update_composition_result(re)
        self._w2._sp.update_camera_actions(re)
        self._st.showMessage("Done | "+a.category.value+" | conf "+str(round(re.overall_confidence*100))+"%")
    def _err(self, m): self._st.showMessage("Warning: "+m)
    def _sw(self, i): self._ws.setCurrentIndex(i)
    def _tog(self, s): self._w2.set_overlay(reverse_eng=s==Qt.CheckState.Checked.value)
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        u=e.mimeData().urls()
        if u:
            p=u[0].toLocalFile()
            if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")): self._la(p)
    def closeEvent(self, e): self._det.close(); e.accept()