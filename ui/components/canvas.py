import math
from typing import List, Dict, Any, Optional
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QPixmap, QPainter, QPen, QBrush, QColor, QFont,
    QWheelEvent, QMouseEvent, QKeyEvent, QCursor
)
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsSimpleTextItem, QGraphicsItem
)

# Color palette for classes
CLASS_COLORS = [
    QColor("#3b82f6"),  # Blue
    QColor("#10b981"),  # Emerald
    QColor("#f59e0b"),  # Amber
    QColor("#8b5cf6"),  # Violet
    QColor("#ec4899"),  # Pink
    QColor("#06b6d4"),  # Cyan
    QColor("#ef4444"),  # Red
    QColor("#84cc16"),  # Lime
]

def get_color_for_label(label: str) -> QColor:
    idx = abs(hash(label.lower())) % len(CLASS_COLORS)
    return CLASS_COLORS[idx]


class AnnotationCanvas(QGraphicsView):
    """
    Interactive canvas for image display and bounding box annotation.
    Supports zoom, pan, drawing, moving, resizing, and label display.
    """
    boxes_changed = Signal()
    box_selected = Signal(int)  # Index of selected box or -1

    MODE_NONE = 0
    MODE_DRAW = 1
    MODE_DRAG = 2
    MODE_RESIZE = 3
    MODE_PAN = 4

    HANDLE_SIZE = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.image_item: Optional[QGraphicsPixmapItem] = None
        self.current_image_path: str = ""
        self.image_width = 1
        self.image_height = 1

        # Boxes list: list of dict {"label": str, "box_2d": [ymin, xmin, ymax, xmax]}
        self.boxes: List[Dict[str, Any]] = []
        self.selected_index: int = -1
        self.current_label: str = "pallet"

        # Interaction state
        self.interaction_mode = self.MODE_NONE
        self.drag_start = QPointF()
        self.temp_rect = QRectF()
        self.active_handle = -1  # 0: TL, 1: TR, 2: BR, 3: BL
        self.is_space_pressed = False

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def load_image(self, image_path: str, boxes: Optional[List[Dict[str, Any]]] = None):
        """Load an image and existing boxes into the canvas."""
        self.scene.clear()
        self.current_image_path = image_path
        self.boxes = list(boxes) if boxes is not None else []
        self.selected_index = -1

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return

        self.image_width = pixmap.width()
        self.image_height = pixmap.height()

        self.image_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(0, 0, self.image_width, self.image_height))

        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self.viewport().update()

    def set_boxes(self, boxes: List[Dict[str, Any]]):
        """Update boxes list."""
        self.boxes = list(boxes)
        self.selected_index = -1
        self.viewport().update()
        self.boxes_changed.emit()

    def set_current_label(self, label: str):
        """Set active label for newly drawn boxes."""
        self.current_label = label.strip() or "object"
        if 0 <= self.selected_index < len(self.boxes):
            self.boxes[self.selected_index]["label"] = self.current_label
            self.viewport().update()
            self.boxes_changed.emit()

    def delete_selected_box(self):
        """Delete currently selected box."""
        if 0 <= self.selected_index < len(self.boxes):
            self.boxes.pop(self.selected_index)
            self.selected_index = -1
            self.viewport().update()
            self.boxes_changed.emit()
            self.box_selected.emit(-1)

    def clear_all_boxes(self):
        """Clear all boxes."""
        self.boxes.clear()
        self.selected_index = -1
        self.viewport().update()
        self.boxes_changed.emit()
        self.box_selected.emit(-1)

    # ---------------- Coordinate Transformations ----------------
    def box_to_scene_rect(self, box_2d: List[int]) -> QRectF:
        """Convert [ymin, xmin, ymax, xmax] (0-1000 scale) to scene QRectF."""
        ymin, xmin, ymax, xmax = box_2d
        x = (xmin / 1000.0) * self.image_width
        y = (ymin / 1000.0) * self.image_height
        w = ((xmax - xmin) / 1000.0) * self.image_width
        h = ((ymax - ymin) / 1000.0) * self.image_height
        return QRectF(x, y, w, h)

    def scene_rect_to_box(self, rect: QRectF) -> List[int]:
        """Convert scene QRectF to [ymin, xmin, ymax, xmax] (0-1000 scale)."""
        norm_rect = rect.normalized()
        xmin = int(round((norm_rect.left() / float(self.image_width)) * 1000.0))
        ymin = int(round((norm_rect.top() / float(self.image_height)) * 1000.0))
        xmax = int(round((norm_rect.right() / float(self.image_width)) * 1000.0))
        ymax = int(round((norm_rect.bottom() / float(self.image_height)) * 1000.0))

        xmin = max(0, min(1000, xmin))
        ymin = max(0, min(1000, ymin))
        xmax = max(0, min(1000, xmax))
        ymax = max(0, min(1000, ymax))
        return [ymin, xmin, ymax, xmax]

    def _get_handles(self, rect: QRectF) -> List[QPointF]:
        """Return 4 corner handle centers."""
        return [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomRight(),
            rect.bottomLeft(),
        ]

    # ---------------- Mouse & Keyboard Events ----------------
    def wheelEvent(self, event: QWheelEvent):
        """Zoom in / zoom out with mouse wheel."""
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            self.is_space_pressed = True
            self.setCursor(Qt.OpenHandCursor)
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected_box()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            self.is_space_pressed = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if not self.image_item:
            return

        scene_pos = self.mapToScene(event.pos())

        # Middle click or Space + Left click -> Pan
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self.is_space_pressed):
            self.interaction_mode = self.MODE_PAN
            self.drag_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton:
            # Check if clicked on a handle of selected box
            if 0 <= self.selected_index < len(self.boxes):
                rect = self.box_to_scene_rect(self.boxes[self.selected_index]["box_2d"])
                handles = self._get_handles(rect)
                for idx, h_pt in enumerate(handles):
                    h_rect = QRectF(h_pt.x() - 8, h_pt.y() - 8, 16, 16)
                    if h_rect.contains(scene_pos):
                        self.interaction_mode = self.MODE_RESIZE
                        self.active_handle = idx
                        self.drag_start = scene_pos
                        self.temp_rect = rect
                        return

            # Check if clicked inside an existing box (from top to bottom)
            clicked_idx = -1
            for idx in reversed(range(len(self.boxes))):
                rect = self.box_to_scene_rect(self.boxes[idx]["box_2d"])
                if rect.contains(scene_pos):
                    clicked_idx = idx
                    break

            if clicked_idx != -1:
                self.selected_index = clicked_idx
                self.interaction_mode = self.MODE_DRAG
                self.drag_start = scene_pos
                self.temp_rect = self.box_to_scene_rect(self.boxes[clicked_idx]["box_2d"])
                self.box_selected.emit(self.selected_index)
                self.viewport().update()
                return

            # Otherwise, start drawing a new box
            self.selected_index = -1
            self.box_selected.emit(-1)
            self.interaction_mode = self.MODE_DRAW
            self.drag_start = scene_pos
            self.temp_rect = QRectF(scene_pos, scene_pos)
            self.viewport().update()

    def mouseMoveEvent(self, event: QMouseEvent):
        scene_pos = self.mapToScene(event.pos())

        if self.interaction_mode == self.MODE_PAN:
            delta = event.pos() - self.drag_start
            self.drag_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return

        if self.interaction_mode == self.MODE_DRAW:
            self.temp_rect = QRectF(self.drag_start, scene_pos).normalized()
            self.viewport().update()
            return

        if self.interaction_mode == self.MODE_DRAG and 0 <= self.selected_index < len(self.boxes):
            dx = scene_pos.x() - self.drag_start.x()
            dy = scene_pos.y() - self.drag_start.y()
            new_rect = self.temp_rect.translated(dx, dy)
            # Clamp inside image boundary
            new_rect = new_rect.intersected(QRectF(0, 0, self.image_width, self.image_height))
            self.boxes[self.selected_index]["box_2d"] = self.scene_rect_to_box(new_rect)
            self.viewport().update()
            return

        if self.interaction_mode == self.MODE_RESIZE and 0 <= self.selected_index < len(self.boxes):
            r = QRectF(self.temp_rect)
            if self.active_handle == 0:  # TL
                r.setTopLeft(scene_pos)
            elif self.active_handle == 1:  # TR
                r.setTopRight(scene_pos)
            elif self.active_handle == 2:  # BR
                r.setBottomRight(scene_pos)
            elif self.active_handle == 3:  # BL
                r.setBottomLeft(scene_pos)

            norm_r = r.normalized().intersected(QRectF(0, 0, self.image_width, self.image_height))
            self.boxes[self.selected_index]["box_2d"] = self.scene_rect_to_box(norm_r)
            self.viewport().update()
            return

        # Cursor change on hover
        if not self.is_space_pressed and 0 <= self.selected_index < len(self.boxes):
            rect = self.box_to_scene_rect(self.boxes[self.selected_index]["box_2d"])
            handles = self._get_handles(rect)
            for idx, h_pt in enumerate(handles):
                h_rect = QRectF(h_pt.x() - 8, h_pt.y() - 8, 16, 16)
                if h_rect.contains(scene_pos):
                    if idx in (0, 2):
                        self.setCursor(Qt.SizeFDiagCursor)
                    else:
                        self.setCursor(Qt.SizeBDiagCursor)
                    return

        if not self.is_space_pressed:
            self.setCursor(Qt.CrossCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.interaction_mode == self.MODE_PAN:
            self.interaction_mode = self.MODE_NONE
            self.setCursor(Qt.OpenHandCursor if self.is_space_pressed else Qt.ArrowCursor)
            return

        if self.interaction_mode == self.MODE_DRAW:
            self.interaction_mode = self.MODE_NONE
            final_rect = self.temp_rect.normalized().intersected(QRectF(0, 0, self.image_width, self.image_height))
            if final_rect.width() > 10 and final_rect.height() > 10:
                box_2d = self.scene_rect_to_box(final_rect)
                new_box = {
                    "label": self.current_label,
                    "box_2d": box_2d,
                }
                self.boxes.append(new_box)
                self.selected_index = len(self.boxes) - 1
                self.box_selected.emit(self.selected_index)
                self.boxes_changed.emit()
            self.viewport().update()
            return

        if self.interaction_mode in (self.MODE_DRAG, self.MODE_RESIZE):
            self.interaction_mode = self.MODE_NONE
            self.boxes_changed.emit()
            self.viewport().update()

    # ---------------- Custom Rendering ----------------
    def drawForeground(self, painter: QPainter, rect: QRectF):
        """Render all bounding boxes, labels, and resize handles directly on scene."""
        if not self.image_item:
            return

        font = QFont("Segoe UI", 10, QFont.Bold)
        painter.setFont(font)

        # Draw existing boxes
        for idx, box in enumerate(self.boxes):
            is_selected = (idx == self.selected_index)
            box_2d = box.get("box_2d", [0, 0, 0, 0])
            label = box.get("label", "object")
            color = get_color_for_label(label)

            r = self.box_to_scene_rect(box_2d)

            # Box fill (transparent)
            fill_color = QColor(color)
            fill_color.setAlpha(60 if is_selected else 30)
            painter.setBrush(QBrush(fill_color))

            # Box border
            pen = QPen(color, 3 if is_selected else 2)
            pen.setStyle(Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(r)

            # Draw Label Tag
            tag_text = f"{label} [{box_2d[0]},{box_2d[1]},{box_2d[2]},{box_2d[3]}]"
            metrics = painter.fontMetrics()
            text_w = metrics.horizontalAdvance(tag_text) + 12
            text_h = metrics.height() + 4

            tag_y = r.top() - text_h
            if tag_y < 0:
                tag_y = r.top() + 2

            tag_rect = QRectF(r.left(), tag_y, text_w, text_h)

            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(tag_rect, 4, 4)

            painter.setPen(QColor("#ffffff"))
            painter.drawText(tag_rect, Qt.AlignCenter, tag_text)

            # If selected, draw corner handles
            if is_selected:
                painter.setBrush(QBrush(QColor("#ffffff")))
                painter.setPen(QPen(color, 2))
                for h_pt in self._get_handles(r):
                    h_rect = QRectF(h_pt.x() - 4, h_pt.y() - 4, 8, 8)
                    painter.drawRect(h_rect)

        # Draw drawing preview box
        if self.interaction_mode == self.MODE_DRAW and self.temp_rect.isValid():
            curr_color = get_color_for_label(self.current_label)
            painter.setBrush(QBrush(QColor(curr_color.red(), curr_color.green(), curr_color.blue(), 40)))
            pen = QPen(curr_color, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.temp_rect)
