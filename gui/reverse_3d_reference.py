"""Small adapter around the reference-aware reconstruction workspace."""
from __future__ import annotations

from gui.reverse_3d_workspace import (
    AnchorProjectionPreview,
    AnchorSceneView,
    CollapsibleSection,
    Reverse3DWorkspace as _BaseReverse3DWorkspace,
)


class Reverse3DWorkspace(_BaseReverse3DWorkspace):
    """Avoid rebuilding the reference hypothesis on every polling tick."""

    def _poll_reference_mode(self):
        window = self.window()
        widget = getattr(window, "_reference_mode", None)
        if widget is None:
            return

        ref = getattr(widget, "_reference", None)
        pose = getattr(widget, "_current_pose", None)
        image = getattr(widget, "_current_image", None)
        signature = (
            id(ref),
            id(pose),
            tuple(image.shape[:2]) if image is not None else None,
        )
        if signature == self._last_ref_signature:
            return
        self._last_ref_signature = signature

        if ref is None or pose is None or image is None:
            self.clear_reference_context()
            return

        from reverse_engineering.reference_reconstruction import build_reference_composition
        if (
            int(getattr(pose, "image_width", 0) or 0) != int(image.shape[1])
            or int(getattr(pose, "image_height", 0) or 0) != int(image.shape[0])
        ):
            pose = pose.rescaled(int(image.shape[1]), int(image.shape[0]))
        current = build_reference_composition(pose, image.shape[1], image.shape[0])
        self.set_reference_context(ref, current)


__all__ = [
    "AnchorProjectionPreview",
    "AnchorSceneView",
    "CollapsibleSection",
    "Reverse3DWorkspace",
]
