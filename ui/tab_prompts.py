import os
import json
from typing import List, Dict, Any, Callable
from PySide6.QtCore import Qt, Signal, QRectF, QTimer
from PySide6.QtGui import (
    QPixmap, QPainter, QPen, QBrush, QColor, QFont
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QSplitter,
    QMessageBox, QFrame, QGroupBox, QComboBox, QSlider,
    QFileDialog, QScrollArea, QCheckBox, QSpinBox, QApplication,
    QDialog, QDoubleSpinBox, QFormLayout, QDialogButtonBox
)
from core.exporter import DatasetExporter
from ui.components.canvas import get_color_for_label


class FrontierScoreDialog(QDialog):
    """Dialog to configure frontier score for negative / exploration prompts."""
    def __init__(self, current_score: float = 0.85, title: str = "Pengaturan Frontier Score", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(360)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        info_lbl = QLabel(
            "<b>Frontier Score (0.00 - 1.00)</b>:<br>"
            "Estimasi seberapa menjanjikan koridor/arah ini untuk menemukan target "
            "(misal 0.85 = lorong sangat mungkin mengarah ke target)."
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
        layout.addWidget(info_lbl)

        form = QFormLayout()
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.00, 1.00)
        self.spin.setSingleStep(0.05)
        self.spin.setDecimals(2)
        self.spin.setValue(current_score)
        form.addRow("Nilai Skor:", self.spin)
        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_score(self) -> float:
        return round(float(self.spin.value()), 2)


class PromptRowWidget(QFrame):
    """Single prompt variation editor row (User Prompt <-> Assistant Response)."""
    delete_requested = Signal(object)
    duplicate_requested = Signal(object)
    changed = Signal()

    def __init__(
        self,
        user_prompt: str = "",
        assistant_response: str = "",
        get_active_boxes: Callable = None,
        get_active_classes: Callable = None,
        parent=None
    ):
        super().__init__(parent)
        self.get_active_boxes = get_active_boxes
        self.get_active_classes = get_active_classes
        self.setObjectName("card")
        self._init_ui(user_prompt, assistant_response)

    def _init_ui(self, user_prompt: str, assistant_response: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Top row: User input
        u_layout = QHBoxLayout()
        u_lbl = QLabel("User Prompt:")
        u_lbl.setFixedWidth(110)
        u_lbl.setStyleSheet("font-weight: 600; color: #38bdf8;")
        self.user_edit = QLineEdit(user_prompt)
        self.user_edit.setPlaceholderText("Contoh: Cari objek 'dispenser' pada citra ini.")
        self.user_edit.textChanged.connect(self.changed.emit)
        u_layout.addWidget(u_lbl)
        u_layout.addWidget(self.user_edit)
        layout.addLayout(u_layout)

        # Bottom row: Assistant output & actions
        a_layout = QHBoxLayout()
        a_lbl = QLabel("Model Output:")
        a_lbl.setFixedWidth(110)
        a_lbl.setStyleSheet("font-weight: 600; color: #10b981;")
        self.assistant_edit = QLineEdit(assistant_response)
        self.assistant_edit.setPlaceholderText(
            '{"target_detected": true, "label": "...", "bounding_box": [...]} atau {"target_detected": false, "frontier_score": 0.85}'
        )
        self.assistant_edit.textChanged.connect(self.changed.emit)
        a_layout.addWidget(a_lbl)
        a_layout.addWidget(self.assistant_edit)

        # Helper Button: Set Box
        self.set_box_btn = QPushButton("🎯 Set Box")
        self.set_box_btn.setToolTip("Otomatis format JSON deteksi objek target dengan bounding box frame ini")
        self.set_box_btn.clicked.connect(self._on_set_box)
        a_layout.addWidget(self.set_box_btn)

        # Helper Button: Set Frontier
        self.set_frontier_btn = QPushButton("🧭 Set Frontier")
        self.set_frontier_btn.setToolTip("Otomatis format JSON negatif (frontier exploration) dengan skor koridor")
        self.set_frontier_btn.clicked.connect(self._on_set_frontier)
        a_layout.addWidget(self.set_frontier_btn)

        # Action Buttons
        dup_btn = QPushButton("📋 Duplikat")
        dup_btn.setFixedWidth(80)
        dup_btn.clicked.connect(lambda: self.duplicate_requested.emit(self))
        a_layout.addWidget(dup_btn)

        del_btn = QPushButton("🗑️")
        del_btn.setObjectName("dangerBtn")
        del_btn.setFixedWidth(36)
        del_btn.clicked.connect(lambda: self.delete_requested.emit(self))
        a_layout.addWidget(del_btn)

        layout.addLayout(a_layout)

    def _on_set_box(self):
        boxes = self.get_active_boxes() if self.get_active_boxes else []
        lbl = "object"
        b_2d = [0, 0, 0, 0]
        if boxes:
            lbl = boxes[0].get("label", "object")
            b_2d = boxes[0].get("box_2d", [0, 0, 0, 0])

        payload = {
            "target_detected": True,
            "label": lbl,
            "bounding_box": b_2d,
            "frontier_score": None,
        }
        self.assistant_edit.setText(json.dumps(payload, ensure_ascii=False))

        if not self.user_edit.text().strip():
            self.user_edit.setText(f"Cari objek '{lbl}' pada citra ini.")

    def _on_set_frontier(self):
        current_score = 0.85
        try:
            p = json.loads(self.assistant_edit.text().strip())
            if isinstance(p, dict) and p.get("frontier_score") is not None:
                current_score = float(p.get("frontier_score"))
        except Exception:
            pass

        dlg = FrontierScoreDialog(current_score=current_score, parent=self)
        if dlg.exec() == QDialog.Accepted:
            score = dlg.get_score()
            payload = {
                "target_detected": False,
                "label": None,
                "bounding_box": None,
                "frontier_score": score,
            }
            self.assistant_edit.setText(json.dumps(payload, ensure_ascii=False))

            # Suggest prompt if empty
            if not self.user_edit.text().strip():
                boxes = self.get_active_boxes() if self.get_active_boxes else []
                present_classes = {b.get("label", "").lower() for b in boxes}
                all_classes = self.get_active_classes() if self.get_active_classes else []
                missing = [c for c in all_classes if c.lower() not in present_classes]
                target_lbl = missing[0] if missing else "dispenser"
                self.user_edit.setText(f"Apakah lorong ini mengarah ke '{target_lbl}'?")

    def get_data(self) -> Dict[str, str]:
        return {
            "user": self.user_edit.text().strip(),
            "assistant": self.assistant_edit.text().strip(),
        }


class ImageBoundingBoxPreview(QWidget):
    """
    Renders the frame image and overlays normalized bounding boxes [ymin, xmin, ymax, xmax] (0-1000)
    with class label badges. Scales smoothly with aspect ratio maintained.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap: QPixmap = None
        self.boxes: List[Dict[str, Any]] = []
        self.show_boxes: bool = True
        self.image_resolution_text: str = ""
        self.setMinimumSize(260, 200)

    def set_frame(self, image_path: str, boxes: List[Dict[str, Any]]):
        self.boxes = boxes or []
        if image_path and os.path.isfile(image_path):
            self.pixmap = QPixmap(image_path)
            if not self.pixmap.isNull():
                self.image_resolution_text = f"{self.pixmap.width()}×{self.pixmap.height()}"
            else:
                self.image_resolution_text = ""
        else:
            self.pixmap = None
            self.image_resolution_text = ""
        self.update()

    def set_show_boxes(self, show: bool):
        self.show_boxes = show
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        bg_rect = self.rect()
        painter.fillRect(bg_rect, QColor("#090d16"))

        if not self.pixmap or self.pixmap.isNull():
            painter.setPen(QColor("#64748b"))
            painter.setFont(QFont("Inter", 10))
            painter.drawText(
                bg_rect,
                Qt.AlignCenter,
                "Pilih sebuah frame di sebelah kiri\nuntuk melihat gambar dan letak bounding box."
            )
            return

        widget_w = self.width()
        widget_h = self.height()
        pix_w = self.pixmap.width()
        pix_h = self.pixmap.height()

        if pix_w <= 0 or pix_h <= 0:
            return

        # Calculate best fit maintaining aspect ratio
        scale = min(widget_w / pix_w, widget_h / pix_h)
        target_w = pix_w * scale
        target_h = pix_h * scale
        offset_x = (widget_w - target_w) / 2.0
        offset_y = (widget_h - target_h) / 2.0

        target_rect = QRectF(offset_x, offset_y, target_w, target_h)
        painter.drawPixmap(target_rect.toRect(), self.pixmap)

        # Draw bounding boxes
        if self.show_boxes and self.boxes:
            for b in self.boxes:
                label = b.get("label", "object")
                b_2d = b.get("box_2d", [0, 0, 0, 0])
                if len(b_2d) != 4:
                    continue

                ymin, xmin, ymax, xmax = b_2d

                bx = offset_x + (xmin / 1000.0) * target_w
                by = offset_y + (ymin / 1000.0) * target_h
                bw = ((xmax - xmin) / 1000.0) * target_w
                bh = ((ymax - ymin) / 1000.0) * target_h

                box_rect = QRectF(bx, by, bw, bh)
                color = get_color_for_label(label)

                # Translucent fill
                fill_color = QColor(color.red(), color.green(), color.blue(), 45)
                painter.fillRect(box_rect, fill_color)

                # Border
                pen = QPen(color, 2.5)
                pen.setJoinStyle(Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawRect(box_rect)

                # Label tag pill
                tag_text = f"{label} {b_2d}"
                font = QFont("Inter", 8, QFont.Bold)
                painter.setFont(font)
                metrics = painter.fontMetrics()
                text_w = metrics.horizontalAdvance(tag_text) + 10
                text_h = metrics.height() + 4

                tag_x = bx
                tag_y = by - text_h
                if tag_y < offset_y:
                    tag_y = by

                tag_rect = QRectF(tag_x, tag_y, text_w, text_h)
                painter.fillRect(tag_rect, color)
                painter.setPen(QColor("#ffffff"))
                painter.drawText(tag_rect, Qt.AlignCenter, tag_text)


class TabPrompts(QWidget):
    """Menu 3: Prompt Template, Data Augmentation, and Dataset Exporter."""
    dataset_exported = Signal(str)

    def __init__(self, dataset_manager, parent=None):
        super().__init__(parent)
        self.dataset_manager = dataset_manager
        self.current_image_name = ""
        self.images_list = []
        self.prompt_rows: List[PromptRowWidget] = []

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # Header Info Card
        header_card = QFrame()
        header_card.setObjectName("card")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(14, 10, 14, 10)

        title_layout = QVBoxLayout()
        title = QLabel("✍️ Variasi Prompt & Ekspor Dataset VLM")
        title.setObjectName("titleLabel")
        subtitle = QLabel("Kustomisasi teks prompt user manual/otomatis, tambahkan variasi dan sampel 'null', lalu ekspor ke JSONL.")
        subtitle.setObjectName("subtitleLabel")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header_layout.addLayout(title_layout)

        header_layout.addStretch()

        self.summary_badge = QLabel("0 Frame")
        self.summary_badge.setObjectName("badgeLabel")
        header_layout.addWidget(self.summary_badge)

        main_layout.addWidget(header_card)

        # Main Splitter (Left: Frame List & Context, Right: Prompt Variations Editor)
        splitter = QSplitter(Qt.Horizontal)

        # ---------------- Left Panel: Frame Selector & Object Info ----------------
        left_panel = QFrame()
        left_panel.setObjectName("panelCard")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        left_title = QLabel("Pilih Frame:")
        left_title.setStyleSheet("font-weight: 700; color: #38bdf8;")
        left_layout.addWidget(left_title)

        self.image_list_widget = QListWidget()
        self.image_list_widget.currentRowChanged.connect(self._on_image_selected)
        left_layout.addWidget(self.image_list_widget)

        # Object Info in selected frame
        info_box = QGroupBox("Objek di Frame Ini")
        info_layout = QVBoxLayout(info_box)
        self.objects_info_label = QLabel("Tidak ada objek terdeteksi.")
        self.objects_info_label.setWordWrap(True)
        self.objects_info_label.setStyleSheet("color: #cbd5e1; font-size: 12px;")
        info_layout.addWidget(self.objects_info_label)
        left_layout.addWidget(info_box)

        # Quick Generation Buttons
        quick_box = QGroupBox("Generator Cepat")
        quick_layout = QVBoxLayout(quick_box)
        quick_layout.setSpacing(8)

        self.gen_all_frames_btn = QPushButton("🚀 Buat Prompt SEMUA Frame")
        self.gen_all_frames_btn.setObjectName("primaryBtn")
        self.gen_all_frames_btn.setFixedHeight(34)
        self.gen_all_frames_btn.setToolTip("Generate otomatis variasi prompt untuk seluruh frame teranotasi di folder ini")
        self.gen_all_frames_btn.clicked.connect(self._auto_generate_all_frames)
        quick_layout.addWidget(self.gen_all_frames_btn)

        self.gen_from_boxes_btn = QPushButton("⚡ Buat untuk Frame Ini Saja")
        self.gen_from_boxes_btn.clicked.connect(self._auto_generate_from_boxes)
        quick_layout.addWidget(self.gen_from_boxes_btn)

        self.gen_negative_all_btn = QPushButton("🧭 Tambah Frontier (Score) SEMUA")
        self.gen_negative_all_btn.setToolTip("Tambah sampel frontier exploration dengan skor koridor untuk semua frame")
        self.gen_negative_all_btn.clicked.connect(self._add_negative_all_frames)
        quick_layout.addWidget(self.gen_negative_all_btn)

        self.gen_negative_btn = QPushButton("🧭 Tambah Frontier Frame Ini")
        self.gen_negative_btn.clicked.connect(self._add_negative_prompt)
        quick_layout.addWidget(self.gen_negative_btn)

        left_layout.addWidget(quick_box)

        left_panel.setMinimumWidth(240)
        left_panel.setMaximumWidth(280)
        splitter.addWidget(left_panel)

        # ---------------- Center Panel: Image Preview with Bounding Boxes ----------------
        preview_panel = QFrame()
        preview_panel.setObjectName("panelCard")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(8)

        # Header preview
        prev_header = QHBoxLayout()
        self.preview_title = QLabel("🖼️ Preview Frame & Bounding Box")
        self.preview_title.setStyleSheet("font-weight: 700; color: #38bdf8; font-size: 13px;")
        prev_header.addWidget(self.preview_title)
        prev_header.addStretch()

        self.res_badge = QLabel("")
        self.res_badge.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 500;")
        prev_header.addWidget(self.res_badge)
        preview_layout.addLayout(prev_header)

        # Preview display widget
        self.image_preview_widget = ImageBoundingBoxPreview()
        preview_layout.addWidget(self.image_preview_widget, 1)

        # Bottom toolbar for preview
        prev_toolbar = QHBoxLayout()
        self.show_box_check = QCheckBox("Tampilkan Bounding Box")
        self.show_box_check.setChecked(True)
        self.show_box_check.toggled.connect(self.image_preview_widget.set_show_boxes)
        prev_toolbar.addWidget(self.show_box_check)

        prev_toolbar.addStretch()

        self.copy_boxes_btn = QPushButton("📋 Salin Box Target")
        self.copy_boxes_btn.setToolTip("Salin koordinat bounding box ke clipboard untuk dipakai di Prompt Model Output")
        self.copy_boxes_btn.clicked.connect(self._copy_boxes_to_clipboard)
        prev_toolbar.addWidget(self.copy_boxes_btn)

        preview_layout.addLayout(prev_toolbar)
        splitter.addWidget(preview_panel)

        # ---------------- Right Panel: Prompt Variations Editor ----------------
        right_panel = QFrame()
        right_panel.setObjectName("panelCard")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)

        editor_header = QHBoxLayout()
        self.editor_title = QLabel("Daftar Variasi Prompt untuk Frame")
        self.editor_title.setStyleSheet("font-weight: 700; color: #38bdf8; font-size: 14px;")
        editor_header.addWidget(self.editor_title)

        editor_header.addStretch()

        self.add_manual_btn = QPushButton("➕ Tambah Prompt Manual")
        self.add_manual_btn.setObjectName("primaryBtn")
        self.add_manual_btn.clicked.connect(self._add_empty_prompt_row)
        editor_header.addWidget(self.add_manual_btn)

        right_layout.addLayout(editor_header)

        # Scrollable area for prompt rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        self.prompt_container = QWidget()
        self.prompt_layout = QVBoxLayout(self.prompt_container)
        self.prompt_layout.setContentsMargins(0, 0, 0, 0)
        self.prompt_layout.setSpacing(8)
        self.prompt_layout.addStretch()

        scroll.setWidget(self.prompt_container)
        right_layout.addWidget(scroll)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        main_layout.addWidget(splitter)

        # ---------------- Bottom Card: Export Settings & Button ----------------
        export_box = QGroupBox("🚀 Ekspor Dataset ke Format Siap Latih (Qwen2-VL / ShareGPT)")
        export_layout = QVBoxLayout(export_box)
        export_layout.setSpacing(10)

        row1 = QHBoxLayout()

        # Format selector
        fmt_lbl = QLabel("Format:")
        self.format_combo = QComboBox()
        self.format_combo.addItem("Qwen2-VL (Grounding + Frontier JSON)", "qwen")
        self.format_combo.addItem("ShareGPT / LLaVA (Grounding + Frontier JSON)", "sharegpt")
        self.format_combo.addItem("Qwen2-VL (Legacy Plain Text)", "qwen_legacy")
        row1.addWidget(fmt_lbl)
        row1.addWidget(self.format_combo)

        # Split Method selector (Anti-Leakage)
        method_lbl = QLabel("Metode Split:")
        self.split_method_combo = QComboBox()
        self.split_method_combo.addItem("Group by Image (Bebas Leakage Prompt)", "image")
        self.split_method_combo.addItem("Sequential Video Chunk (Bebas Leakage Video)", "sequential")
        self.split_method_combo.setToolTip(
            "Group by Image: Memastikan semua variasi prompt dari 1 gambar masuk ke set yang sama.\n"
            "Sequential: Membagi frame secara kronologis berurutan (mencegah frame video mirip terpecah antara Train dan Val)."
        )
        row1.addWidget(method_lbl)
        row1.addWidget(self.split_method_combo)

        # Train / Val split
        split_lbl = QLabel("Train Ratio:")
        self.split_slider = QSlider(Qt.Horizontal)
        self.split_slider.setRange(50, 95)
        self.split_slider.setValue(80)
        self.split_value_lbl = QLabel("80% Train / 20% Val")
        self.split_value_lbl.setFixedWidth(130)
        self.split_slider.valueChanged.connect(
            lambda v: self.split_value_lbl.setText(f"{v}% Train / {100 - v}% Val")
        )

        row1.addWidget(split_lbl)
        row1.addWidget(self.split_slider)
        row1.addWidget(self.split_value_lbl)

        # Copy images checkbox
        self.copy_images_check = QCheckBox("Salin Gambar")
        self.copy_images_check.setChecked(True)
        self.copy_images_check.setToolTip("Salin file gambar ke subfolder 'images' di folder tujuan export.")
        row1.addWidget(self.copy_images_check)

        export_layout.addLayout(row1)

        # Output folder row
        row2 = QHBoxLayout()
        out_lbl = QLabel("Folder Tujuan Export:")
        out_lbl.setFixedWidth(140)
        self.export_dir_edit = QLineEdit()
        self.export_dir_edit.setPlaceholderText("Pilih folder output export...")
        self.browse_export_btn = QPushButton("Pilih Folder...")
        self.browse_export_btn.clicked.connect(self._browse_export_dir)

        self.export_btn = QPushButton("✨ Ekspor Dataset (JSONL)")
        self.export_btn.setObjectName("successBtn")
        self.export_btn.setFixedHeight(36)
        self.export_btn.clicked.connect(self._run_export)

        row2.addWidget(out_lbl)
        row2.addWidget(self.export_dir_edit)
        row2.addWidget(self.browse_export_btn)
        row2.addWidget(self.export_btn)

        export_layout.addLayout(row2)

        main_layout.addWidget(export_box)

    # ---------------- Data Loading & Sync ----------------
    def reload_data(self):
        """Reload image list and refresh views."""
        self.images_list = self.dataset_manager.get_image_list()
        self.image_list_widget.blockSignals(True)
        self.image_list_widget.clear()

        annotated_count = 0
        first_annotated_idx = 0
        found_first_ann = False

        for idx, img in enumerate(self.images_list):
            anno = self.dataset_manager.get_annotation(img)
            b_count = len(anno.get("boxes", []))
            p_count = len(anno.get("prompts", []))
            if b_count > 0:
                annotated_count += 1
                if not found_first_ann:
                    first_annotated_idx = idx
                    found_first_ann = True

            icon = "🟢" if b_count > 0 else "⚪"
            label = f"{icon} [{b_count} box, {p_count} prompt] {img}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, img)
            self.image_list_widget.addItem(item)

        self.image_list_widget.blockSignals(False)

        folder_name = os.path.basename(self.dataset_manager.active_frames_dir or "frames")
        self.summary_badge.setText(f"📁 {folder_name} | {len(self.images_list)} Frame ({annotated_count} Teranotasi)")

        # Set default export dir if empty
        if not self.export_dir_edit.text() and self.dataset_manager.project_dir:
            self.export_dir_edit.setText(os.path.join(self.dataset_manager.project_dir, "exports"))

        if self.image_list_widget.count() > 0:
            # Auto-select the first annotated frame so user sees detected objects right away
            self.image_list_widget.setCurrentRow(first_annotated_idx)
        else:
            self.current_image_name = ""
            self.editor_title.setText("Daftar Variasi Prompt")
            self.preview_title.setText("🖼️ Preview Frame & Bounding Box")
            self.res_badge.setText("")
            self.image_preview_widget.set_frame("", [])
            self.objects_info_label.setText("Tidak ada gambar dalam folder ini.")
            self._clear_prompt_rows()

    def _on_image_selected(self, row: int):
        if row < 0:
            return
        item = self.image_list_widget.item(row)
        if not item:
            return
        img_name = item.data(Qt.UserRole)
        self._load_prompts_for_image(img_name)

    def _load_prompts_for_image(self, img_name: str):
        # Save previous image before switching
        if self.current_image_name and self.current_image_name != img_name:
            self._save_current_prompts()

        self.current_image_name = img_name
        self.editor_title.setText(f"Variasi Prompt untuk: {img_name}")
        self.preview_title.setText(f"🖼️ {img_name}")

        anno = self.dataset_manager.get_annotation(img_name)
        boxes = anno.get("boxes", [])
        prompts = anno.get("prompts", [])

        # Update preview frame & boxes
        img_path = self.dataset_manager.get_image_path(img_name)
        self.image_preview_widget.set_frame(img_path, boxes)
        res_text = self.image_preview_widget.image_resolution_text
        box_count_text = f"{len(boxes)} Box" if boxes else "0 Box (Negatif)"
        if res_text:
            self.res_badge.setText(f"{res_text} • {box_count_text}")
        else:
            self.res_badge.setText(box_count_text)

        # Update object info box
        if boxes:
            obj_lines = [f"• <b>{b.get('label', 'obj')}</b>: {b.get('box_2d', [])}" for b in boxes]
            self.objects_info_label.setText("<br>".join(obj_lines))
        else:
            self.objects_info_label.setText("<i>Tidak ada objek di frame ini (Bagus untuk negative sample).</i>")

        # Populate prompt rows
        self._clear_prompt_rows()
        for p in prompts:
            self._add_prompt_row_ui(p.get("user", ""), p.get("assistant", ""))

    def _copy_boxes_to_clipboard(self):
        if not self.current_image_name:
            return
        anno = self.dataset_manager.get_annotation(self.current_image_name)
        boxes = anno.get("boxes", [])
        if not boxes:
            payload = {
                "target_detected": False,
                "label": None,
                "bounding_box": None,
                "frontier_score": 0.85,
            }
            json_str = json.dumps(payload, ensure_ascii=False)
            QApplication.clipboard().setText(json_str)
            self.copy_boxes_btn.setText("✅ Disalin (Frontier)")
        else:
            b0 = boxes[0]
            payload = {
                "target_detected": True,
                "label": b0.get("label", "object"),
                "bounding_box": b0.get("box_2d", [0, 0, 0, 0]),
                "frontier_score": None,
            }
            json_str = json.dumps(payload, ensure_ascii=False)
            QApplication.clipboard().setText(json_str)
            self.copy_boxes_btn.setText("✅ Disalin (Box JSON)!")

        QTimer.singleShot(1500, lambda: self.copy_boxes_btn.setText("📋 Salin Box Target"))

    def _get_current_boxes(self):
        if not self.current_image_name:
            return []
        anno = self.dataset_manager.get_annotation(self.current_image_name)
        return anno.get("boxes", [])

    def _clear_prompt_rows(self):
        for row in self.prompt_rows:
            self.prompt_layout.removeWidget(row)
            row.deleteLater()
        self.prompt_rows.clear()

    def _add_prompt_row_ui(self, user_text: str = "", assistant_text: str = ""):
        if not assistant_text:
            payload = {
                "target_detected": False,
                "label": None,
                "bounding_box": None,
                "frontier_score": 0.85,
            }
            assistant_text = json.dumps(payload, ensure_ascii=False)

        row = PromptRowWidget(
            user_prompt=user_text,
            assistant_response=assistant_text,
            get_active_boxes=self._get_current_boxes,
            get_active_classes=lambda: self.dataset_manager.classes,
            parent=self.prompt_container,
        )
        row.delete_requested.connect(self._on_delete_row)
        row.duplicate_requested.connect(self._on_duplicate_row)
        row.changed.connect(self._save_current_prompts)

        # Insert before the last stretch
        idx = max(0, self.prompt_layout.count() - 1)
        self.prompt_layout.insertWidget(idx, row)
        self.prompt_rows.append(row)

    def _add_empty_prompt_row(self):
        if not self.current_image_name:
            return
        boxes = self._get_current_boxes()
        if boxes:
            lbl = boxes[0].get("label", "object")
            b_2d = boxes[0].get("box_2d", [0, 0, 0, 0])
            payload = {
                "target_detected": True,
                "label": lbl,
                "bounding_box": b_2d,
                "frontier_score": None,
            }
            u_text = f"Cari objek '{lbl}' pada citra ini."
            a_text = json.dumps(payload, ensure_ascii=False)
        else:
            payload = {
                "target_detected": False,
                "label": None,
                "bounding_box": None,
                "frontier_score": 0.85,
            }
            u_text = "Cari objek 'dispenser' di koridor ini."
            a_text = json.dumps(payload, ensure_ascii=False)

        self._add_prompt_row_ui(u_text, a_text)
        self._save_current_prompts()

    def _on_delete_row(self, row_widget):
        if row_widget in self.prompt_rows:
            self.prompt_rows.remove(row_widget)
            self.prompt_layout.removeWidget(row_widget)
            row_widget.deleteLater()
            self._save_current_prompts()

    def _on_duplicate_row(self, row_widget):
        data = row_widget.get_data()
        self._add_prompt_row_ui(data.get("user", ""), data.get("assistant", ""))
        self._save_current_prompts()

    def _save_current_prompts(self):
        if not self.current_image_name:
            return

        anno = self.dataset_manager.get_annotation(self.current_image_name)
        new_prompts = []
        for idx, row in enumerate(self.prompt_rows):
            d = row.get_data()
            if d.get("user") or d.get("assistant"):
                new_prompts.append({
                    "id": f"p_{idx+1}",
                    "user": d.get("user", ""),
                    "assistant": d.get("assistant", ""),
                })

        anno["prompts"] = new_prompts
        self.dataset_manager.save_annotation(self.current_image_name, anno)

        # Update row item text in list
        curr_row = self.image_list_widget.currentRow()
        if curr_row >= 0:
            item = self.image_list_widget.item(curr_row)
            b_count = len(anno.get("boxes", []))
            p_count = len(new_prompts)
            icon = "🟢" if b_count > 0 else "⚪"
            item.setText(f"{icon} [{b_count} box, {p_count} prompt] {self.current_image_name}")

    # ---------------- Quick Generators ----------------
    def _auto_generate_all_frames(self):
        """Generate prompt variations for ALL annotated frames in active folder with Semantic Grounding JSON."""
        images = self.dataset_manager.get_image_list()
        if not images:
            QMessageBox.information(self, "Info", "Tidak ada gambar di folder aktif.")
            return

        variations_per_box = [
            "Cari objek '{label}' pada citra ini.",
            "Tunjukkan lokasi '{label}' di ruangan ini.",
            "Deteksi '{label}' di depan robot.",
        ]

        total_frames_processed = 0
        total_prompts_created = 0

        for img_name in images:
            anno = self.dataset_manager.get_annotation(img_name)
            boxes = anno.get("boxes", [])
            if not boxes:
                continue

            existing_prompts = anno.get("prompts", [])
            new_prompts = list(existing_prompts)

            for b in boxes:
                lbl = b.get("label", "object")
                b_2d = b.get("box_2d", [0, 0, 0, 0])
                payload = {
                    "target_detected": True,
                    "label": lbl,
                    "bounding_box": b_2d,
                    "frontier_score": None,
                }
                assistant_json = json.dumps(payload, ensure_ascii=False)

                for template in variations_per_box:
                    u_text = template.format(label=lbl)
                    if not any(p.get("user") == u_text for p in new_prompts):
                        new_prompts.append({
                            "id": f"p_{len(new_prompts)+1}",
                            "user": u_text,
                            "assistant": assistant_json,
                        })
                        total_prompts_created += 1

            anno["prompts"] = new_prompts
            self.dataset_manager.save_annotation(img_name, anno)
            total_frames_processed += 1

        if total_frames_processed == 0:
            QMessageBox.warning(
                self,
                "Tidak Ada Bounding Box",
                "Tidak ditemukan frame dengan bounding box di folder ini.\n"
                "Silakan kembali ke Menu 2 untuk menganotasi objek terlebih dahulu."
            )
            return

        self.reload_data()
        QMessageBox.information(
            self,
            "Batch Prompt Sukses! 🎉",
            f"Berhasil membuat {total_prompts_created} variasi prompt untuk {total_frames_processed} frame teranotasi!\n\n"
            "Format output: Semantic Grounding JSON siap latih."
        )

    def _add_negative_all_frames(self):
        """Add frontier exploration prompts with score for all annotated frames."""
        images = self.dataset_manager.get_image_list()
        if not images:
            QMessageBox.information(self, "Info", "Tidak ada gambar di folder aktif.")
            return

        dlg = FrontierScoreDialog(
            current_score=0.85,
            title="Frontier Score untuk Semua Frame",
            parent=self
        )
        if dlg.exec() != QDialog.Accepted:
            return

        score = dlg.get_score()
        count = 0

        templates = [
            "Apakah lorong ini mengarah ke '{label}'?",
            "Cari objek '{label}' di koridor ini.",
            "Find the '{label}' in this image.",
        ]

        payload = {
            "target_detected": False,
            "label": None,
            "bounding_box": None,
            "frontier_score": score,
        }
        assistant_json = json.dumps(payload, ensure_ascii=False)

        for img_name in images:
            anno = self.dataset_manager.get_annotation(img_name)
            boxes = anno.get("boxes", [])
            if not boxes:
                continue

            present_classes = {b.get("label", "").lower() for b in boxes}
            missing = [c for c in self.dataset_manager.classes if c.lower() not in present_classes]
            candidate_label = missing[0] if missing else "obstacle"

            u_text = templates[0].format(label=candidate_label)
            existing = anno.get("prompts", [])
            if not any(p.get("user") == u_text for p in existing):
                existing.append({
                    "id": f"p_{len(existing)+1}",
                    "user": u_text,
                    "assistant": assistant_json,
                })
                anno["prompts"] = existing
                self.dataset_manager.save_annotation(img_name, anno)
                count += 1

        self.reload_data()
        QMessageBox.information(
            self,
            "Frontier Prompt Ditambahkan",
            f"Berhasil menambahkan {count} sampel frontier exploration (Score: {score}) ke seluruh frame teranotasi."
        )

    def _auto_generate_from_boxes(self):
        """Generate positive prompt pairs from bounding boxes with varied questions for CURRENT frame."""
        if not self.current_image_name:
            return

        anno = self.dataset_manager.get_annotation(self.current_image_name)
        boxes = anno.get("boxes", [])
        if not boxes:
            QMessageBox.information(
                self,
                "Belum Ada Objek",
                f"Frame '{self.current_image_name}' belum memiliki bounding box.\n"
                "Silakan pilih frame bertanda hijau 🟢 atau klik '🚀 Buat Prompt SEMUA Frame'."
            )
            return

        variations_per_box = [
            "Cari objek '{label}' pada citra ini.",
            "Tunjukkan lokasi '{label}' di ruangan ini.",
            "Deteksi '{label}' di depan robot.",
        ]

        for b in boxes:
            lbl = b.get("label", "object")
            b_2d = b.get("box_2d", [0, 0, 0, 0])
            payload = {
                "target_detected": True,
                "label": lbl,
                "bounding_box": b_2d,
                "frontier_score": None,
            }
            assistant_json = json.dumps(payload, ensure_ascii=False)

            for template in variations_per_box:
                u_text = template.format(label=lbl)
                self._add_prompt_row_ui(u_text, assistant_json)

        self._save_current_prompts()
        QMessageBox.information(self, "Sukses", f"Berhasil membuat variasi prompt dari {len(boxes)} objek pada frame ini.")

    def _add_negative_prompt(self):
        """Add frontier exploration prompt for a class that does NOT exist in current frame."""
        if not self.current_image_name:
            return

        dlg = FrontierScoreDialog(
            current_score=0.85,
            title="Frontier Score Frame Ini",
            parent=self
        )
        if dlg.exec() != QDialog.Accepted:
            return

        score = dlg.get_score()
        anno = self.dataset_manager.get_annotation(self.current_image_name)
        present_classes = {b.get("label", "").lower() for b in anno.get("boxes", [])}

        missing = [c for c in self.dataset_manager.classes if c.lower() not in present_classes]
        candidate_label = missing[0] if missing else "obstacle"

        u_text = f"Apakah lorong ini mengarah ke '{candidate_label}'?"
        payload = {
            "target_detected": False,
            "label": None,
            "bounding_box": None,
            "frontier_score": score,
        }
        assistant_json = json.dumps(payload, ensure_ascii=False)

        self._add_prompt_row_ui(u_text, assistant_json)
        self._save_current_prompts()


    # ---------------- Export Function ----------------
    def _browse_export_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Pilih Folder Tujuan Export")
        if d:
            self.export_dir_edit.setText(d)

    def _run_export(self):
        self._save_current_prompts()

        out_dir = self.export_dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Peringatan", "Tentukan folder tujuan export.")
            return

        fmt = self.format_combo.currentData()
        method = self.split_method_combo.currentData() or "image"
        ratio = self.split_slider.value() / 100.0
        copy_img = self.copy_images_check.isChecked()

        success, msg, stats = DatasetExporter.export(
            dataset_manager=self.dataset_manager,
            output_dir=out_dir,
            format_type=fmt,
            train_ratio=ratio,
            split_method=method,
            copy_images=copy_img,
        )

        if success:
            summary = (
                f"{msg}\n\n"
                f"• Total Gambar: {stats.get('total_images')}\n"
                f"  - Train: {stats.get('train_images')} gambar ({stats.get('train_samples')} sampel)\n"
                f"  - Val: {stats.get('val_images')} gambar ({stats.get('val_samples')} sampel)\n"
                f"• Positif (Ada Objek): {stats.get('positive_samples')}\n"
                f"• Negatif ('null'): {stats.get('negative_samples_null')}\n"
                f"• Bebas Kebocoran: Gambar di Train dan Val dijamin 100% terpisah!\n\n"
                f"File tersimpan di:\n{out_dir}/train.jsonl\n{out_dir}/val.jsonl"
            )
            QMessageBox.information(self, "Export Berhasil! 🎉", summary)
            self.dataset_exported.emit(out_dir)
        else:
            QMessageBox.warning(self, "Export Gagal", msg)
