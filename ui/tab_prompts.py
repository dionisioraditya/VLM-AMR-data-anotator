import os
import json
from typing import List, Dict, Any, Callable, Optional
from PySide6.QtCore import Qt, Signal, QRectF, QTimer
from PySide6.QtGui import (
    QPixmap, QPainter, QPen, QBrush, QColor, QFont
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QSplitter,
    QMessageBox, QFrame, QGroupBox, QComboBox, QSlider,
    QFileDialog, QScrollArea, QCheckBox, QSpinBox, QApplication,
    QDialog, QDoubleSpinBox, QFormLayout, QDialogButtonBox, QRadioButton
)
from core.exporter import DatasetExporter
from core.prompt_generator import TemplatePromptGenerator
from ui.components.canvas import get_color_for_label
from ui.components.template_manager_dialog import TemplateManagerDialog


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


class FastPromptGeneratorDialog(QDialog):
    """Dialog for fast template-based prompt generation for positive & negative grounding."""

    def __init__(
        self,
        dataset_manager,
        current_image_name: str,
        initial_settings: Optional[Dict[str, Any]] = None,
        parent=None
    ):
        super().__init__(parent)
        self.dataset_manager = dataset_manager
        self.current_image_name = current_image_name
        self.initial_settings = initial_settings or {}

        self.setWindowTitle("⚡ Fast Prompt Generator (Template-Based)")
        self.setFixedWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("⚡ Fast Prompt Generator (Template-Based)")
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        desc = QLabel(
            "Otomatis membuat prompt dari template terisolasi:<br>"
            "• <b>Objek Terdeteksi</b>: Output bounding box asli + <code>frontier_score</code> (dinamis berdasarkan ukuran BBOX atau statis)<br>"
            "• <b>Objek Tidak Ada</b>: Output eksplorasi koridor (<code>target_detected: false</code>, <code>frontier_score: X</code>)"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #cbd5e1; font-size: 12px; line-height: 1.4;")
        layout.addWidget(desc)

        form = QFormLayout()
        form.setSpacing(10)

        # 1. Pilihan Bahasa
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("🇮🇩 Bahasa Indonesia (50 template)", "id")
        self.lang_combo.addItem("🇬🇧 English (50 template)", "en")
        self.lang_combo.addItem("🌐 Bilingual (id + en - 100 template)", "both")

        pref_lang = self.initial_settings.get("language", "both")
        for i in range(self.lang_combo.count()):
            if self.lang_combo.itemData(i) == pref_lang:
                self.lang_combo.setCurrentIndex(i)
                break
        form.addRow("Pilihan Bahasa:", self.lang_combo)

        # 2. Template Quantity
        qty_layout = QHBoxLayout()
        self.all_tpl_radio = QRadioButton("Semua Template")
        self.limit_tpl_radio = QRadioButton("Acak N template per kelas:")
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(1, 100)

        pref_use_limit = self.initial_settings.get("use_limit", False)
        if self.initial_settings.get("max_per_class") is not None:
            pref_use_limit = True

        pref_limit_val = (
            self.initial_settings.get("max_per_class")
            or self.initial_settings.get("limit_spin_value")
            or 10
        )
        self.limit_spin.setValue(int(pref_limit_val))

        if pref_use_limit:
            self.limit_tpl_radio.setChecked(True)
            self.limit_spin.setEnabled(True)
        else:
            self.all_tpl_radio.setChecked(True)
            self.limit_spin.setEnabled(False)

        self.limit_tpl_radio.toggled.connect(self.limit_spin.setEnabled)

        qty_layout.addWidget(self.all_tpl_radio)
        qty_layout.addWidget(self.limit_tpl_radio)
        qty_layout.addWidget(self.limit_spin)
        form.addRow("Kuantitas Template:", qty_layout)

        # 3. Frontier Score Settings (Dynamic, Static, or Null)
        self.frontier_mode_combo = QComboBox()
        self.frontier_mode_combo.addItem("⚡ Dinamis (Otomatis dari ukuran BBOX)", "dynamic")
        self.frontier_mode_combo.addItem("🔒 Statis (Nilai tetap untuk semua objek)", "static")
        self.frontier_mode_combo.addItem("🚫 Null / None (Objek Terdeteksi = null)", "null")

        pref_mode = self.initial_settings.get("frontier_mode", "dynamic")
        for i in range(self.frontier_mode_combo.count()):
            if self.frontier_mode_combo.itemData(i) == pref_mode:
                self.frontier_mode_combo.setCurrentIndex(i)
                break
        form.addRow("Mode Frontier (Objek):", self.frontier_mode_combo)

        # Static Positive Score
        self.pos_score_spin = QDoubleSpinBox()
        self.pos_score_spin.setRange(0.00, 1.00)
        self.pos_score_spin.setSingleStep(0.05)
        self.pos_score_spin.setDecimals(2)
        pref_pos_score = self.initial_settings.get("positive_frontier_score", 0.85)
        try:
            self.pos_score_spin.setValue(float(pref_pos_score))
        except (ValueError, TypeError):
            self.pos_score_spin.setValue(0.85)
        self.pos_score_spin.setToolTip("Nilai frontier score tetap untuk seluruh objek yang terdeteksi (Mode Statis)")

        self.pos_score_label = QLabel("Skor Statis (Objek Positif):")
        form.addRow(self.pos_score_label, self.pos_score_spin)

        def _on_frontier_mode_changed(idx):
            is_static = (self.frontier_mode_combo.itemData(idx) == "static")
            self.pos_score_spin.setEnabled(is_static)
            self.pos_score_label.setEnabled(is_static)

        self.frontier_mode_combo.currentIndexChanged.connect(_on_frontier_mode_changed)
        _on_frontier_mode_changed(self.frontier_mode_combo.currentIndex())

        # Negative Frontier Score (for exploration / non-present classes)
        self.score_spin = QDoubleSpinBox()
        self.score_spin.setRange(0.00, 1.00)
        self.score_spin.setSingleStep(0.05)
        self.score_spin.setDecimals(2)
        pref_score = self.initial_settings.get("frontier_score", 0.85)
        try:
            self.score_spin.setValue(float(pref_score))
        except (ValueError, TypeError):
            self.score_spin.setValue(0.85)
        self.score_spin.setToolTip("Skor koridor eksplorasi untuk kelas yang tidak terdeteksi di gambar (Negatif)")
        form.addRow("Skor Koridor (Objek Negatif):", self.score_spin)

        layout.addLayout(form)

        # 4. Scope
        scope_group = QGroupBox("Cakupan Pembuatan Prompt")
        scope_layout = QVBoxLayout(scope_group)
        self.scope_all_radio = QRadioButton("Seluruh Frame di Folder Aktif (Termasuk Frame Negatif Tanpa Objek)")
        curr_text = f"Hanya Frame Ini Saja ({self.current_image_name})" if self.current_image_name else "Hanya Frame Ini Saja (Belum ada frame dipilih)"
        self.scope_curr_radio = QRadioButton(curr_text)

        pref_scope = self.initial_settings.get("scope", "current" if self.current_image_name else "all")
        if pref_scope == "current" and self.current_image_name:
            self.scope_curr_radio.setChecked(True)
        else:
            self.scope_all_radio.setChecked(True)

        if not self.current_image_name:
            self.scope_curr_radio.setEnabled(False)

        scope_layout.addWidget(self.scope_all_radio)
        scope_layout.addWidget(self.scope_curr_radio)
        layout.addWidget(scope_group)

        # 5. Clear existing
        self.clear_existing_check = QCheckBox("Hapus prompt lama sebelum generate (Replace)")
        pref_clear = self.initial_settings.get("clear_existing", False)
        self.clear_existing_check.setChecked(bool(pref_clear))
        self.clear_existing_check.setStyleSheet("color: #f59e0b;")
        layout.addWidget(self.clear_existing_check)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.generate_btn = QPushButton("▶ Mulai Generate")
        self.generate_btn.setObjectName("primaryBtn")
        self.generate_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.generate_btn)

        self.cancel_btn = QPushButton("Batal")
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)

        layout.addLayout(btn_row)

    def get_settings(self) -> Dict[str, Any]:
        use_limit = self.limit_tpl_radio.isChecked()
        return {
            "language": self.lang_combo.currentData(),
            "use_limit": use_limit,
            "max_per_class": self.limit_spin.value() if use_limit else None,
            "limit_spin_value": self.limit_spin.value(),
            "frontier_mode": self.frontier_mode_combo.currentData(),
            "positive_frontier_score": round(float(self.pos_score_spin.value()), 2),
            "frontier_score": round(float(self.score_spin.value()), 2),
            "scope": "all" if self.scope_all_radio.isChecked() else "current",
            "clear_existing": self.clear_existing_check.isChecked(),
        }


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
    """Menu 4: Segmented Prompt Editor (Train, Validation, Test) with Isolated Template Generation."""
    dataset_updated = Signal()

    def __init__(self, dataset_manager, parent=None):
        super().__init__(parent)
        self.dataset_manager = dataset_manager
        self.active_split = "train"
        self.current_image_name = ""
        self.images_list = []
        self.prompt_rows: List[PromptRowWidget] = []

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(10)

        # 1. Top Card: Split Selector Bar (Train / Val / Test) + Manage Templates
        split_nav_card = QFrame()
        split_nav_card.setObjectName("panelCard")
        split_nav_layout = QHBoxLayout(split_nav_card)
        split_nav_layout.setContentsMargins(14, 10, 14, 10)
        split_nav_layout.setSpacing(10)

        split_lbl = QLabel("Pilih Partisi Dataset:")
        split_lbl.setStyleSheet("font-weight: 700; color: #38bdf8; font-size: 13px;")
        split_nav_layout.addWidget(split_lbl)

        self.btn_split_train = QPushButton("🟢 Train Set (0 Frame)")
        self.btn_split_train.setFixedHeight(34)
        self.btn_split_train.clicked.connect(lambda: self._switch_split("train"))
        split_nav_layout.addWidget(self.btn_split_train)

        self.btn_split_val = QPushButton("🟡 Validation Set (0 Frame)")
        self.btn_split_val.setFixedHeight(34)
        self.btn_split_val.clicked.connect(lambda: self._switch_split("val"))
        split_nav_layout.addWidget(self.btn_split_val)

        self.btn_split_test = QPushButton("🔵 Test Set (0 Frame)")
        self.btn_split_test.setFixedHeight(34)
        self.btn_split_test.clicked.connect(lambda: self._switch_split("test"))
        split_nav_layout.addWidget(self.btn_split_test)

        split_nav_layout.addStretch()

        self.summary_badge = QLabel("Partisi: TRAIN (0 Frame)")
        self.summary_badge.setObjectName("badgeLabel")
        split_nav_layout.addWidget(self.summary_badge)

        self.btn_manage_templates = QPushButton("⚙️ Kelola Template (Train)")
        self.btn_manage_templates.setObjectName("primaryBtn")
        self.btn_manage_templates.setFixedHeight(34)
        self.btn_manage_templates.clicked.connect(self._open_template_manager)
        split_nav_layout.addWidget(self.btn_manage_templates)

        main_layout.addWidget(split_nav_card)

        # 2. Middle Card: Pilih Frame (Horizontal 3-Column Card)
        frame_card = QFrame()
        frame_card.setObjectName("card")
        frame_card.setFixedHeight(195)
        frame_card_layout = QVBoxLayout(frame_card)
        frame_card_layout.setContentsMargins(12, 8, 12, 10)
        frame_card_layout.setSpacing(6)

        frame_card_title = QLabel("Pilih Frame:")
        frame_card_title.setStyleSheet("font-weight: 700; color: #38bdf8; font-size: 13px;")
        frame_card_layout.addWidget(frame_card_title)

        cols_layout = QHBoxLayout()
        cols_layout.setSpacing(10)

        # Sub-kolom 1: Daftar Frame (Lebar)
        self.image_list_widget = QListWidget()
        self.image_list_widget.currentRowChanged.connect(self._on_image_selected)
        cols_layout.addWidget(self.image_list_widget, 3)

        # Sub-kolom 2: Objek di Frame Ini
        info_box = QGroupBox("Objek di Frame Ini")
        info_layout = QVBoxLayout(info_box)
        info_layout.setContentsMargins(8, 6, 8, 6)

        info_scroll = QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QFrame.NoFrame)
        info_scroll.setStyleSheet("background: transparent; border: none;")

        self.objects_info_label = QLabel("Tidak ada objek terdeteksi.")
        self.objects_info_label.setWordWrap(True)
        self.objects_info_label.setStyleSheet("color: #cbd5e1; font-size: 12px; background: transparent;")
        self.objects_info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_scroll.setWidget(self.objects_info_label)
        info_layout.addWidget(info_scroll)

        cols_layout.addWidget(info_box, 2)

        # Sub-kolom 3: Generator Cepat
        quick_box = QGroupBox("Generator Cepat")
        quick_layout = QVBoxLayout(quick_box)
        quick_layout.setContentsMargins(8, 6, 8, 6)
        quick_layout.setSpacing(6)

        self.generator_mode_combo = QComboBox()
        self.generator_mode_combo.addItem("⚡ Fast Generate dari Template", "fast_template")
        self.generator_mode_combo.addItem("🚀 Buat Prompt SEMUA Frame (3 Variasi)", "all_frames")
        self.generator_mode_combo.addItem("⚡ Buat untuk Frame Ini Saja (3 Variasi)", "current_frame")
        self.generator_mode_combo.addItem("🧭 Tambah Frontier (Score) SEMUA", "frontier_all")
        self.generator_mode_combo.addItem("🧭 Tambah Frontier Frame Ini", "frontier_current")
        quick_layout.addWidget(self.generator_mode_combo)

        self.execute_generator_btn = QPushButton("▶ Mulai Generate")
        self.execute_generator_btn.setObjectName("primaryBtn")
        self.execute_generator_btn.setFixedHeight(30)
        self.execute_generator_btn.setToolTip("Jalankan mode generator prompt yang dipilih di atas")
        self.execute_generator_btn.clicked.connect(self._on_execute_generator_clicked)
        quick_layout.addWidget(self.execute_generator_btn)

        self.clear_all_dataset_prompts_btn = QPushButton("🗑️ Hapus SEMUA Prompt Split Ini")
        self.clear_all_dataset_prompts_btn.setObjectName("dangerBtn")
        self.clear_all_dataset_prompts_btn.setFixedHeight(28)
        self.clear_all_dataset_prompts_btn.setToolTip("Hapus seluruh variasi prompt dari frame di split aktif ini")
        self.clear_all_dataset_prompts_btn.clicked.connect(self._clear_all_dataset_prompts)
        quick_layout.addWidget(self.clear_all_dataset_prompts_btn)

        cols_layout.addWidget(quick_box, 2)

        frame_card_layout.addLayout(cols_layout)
        main_layout.addWidget(frame_card)

        # 3. Bottom Work Area: Preview Frame & Prompt Editor (Splitter 2-Kolom)
        splitter = QSplitter(Qt.Horizontal)

        # ---------------- Panel Kiri: Image Preview with Bounding Boxes ----------------
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

        # ---------------- Panel Kanan: Prompt Variations Editor ----------------
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

        self.clear_all_prompts_btn = QPushButton("🗑️ Hapus Semua Prompt")
        self.clear_all_prompts_btn.setObjectName("dangerBtn")
        self.clear_all_prompts_btn.setToolTip("Hapus semua variasi prompt pada frame ini")
        self.clear_all_prompts_btn.clicked.connect(self._clear_all_current_prompts)
        editor_header.addWidget(self.clear_all_prompts_btn)

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
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, 1)

    # ---------------- Split Navigation & Data Loading ----------------
    def _update_split_buttons_ui(self):
        splits = self.dataset_manager.get_splits()
        n_tr = len(splits.get("train", []))
        n_va = len(splits.get("val", []))
        n_te = len(splits.get("test", []))

        self.btn_split_train.setText(f"🟢 Train ({n_tr} Frame)")
        self.btn_split_val.setText(f"🟡 Validation ({n_va} Frame)")
        self.btn_split_test.setText(f"🔵 Test ({n_te} Frame)")

        active = self.active_split
        def style_btn(btn, is_active, color):
            if is_active:
                btn.setStyleSheet(f"background-color: {color}25; color: {color}; border: 2px solid {color}; font-weight: bold; border-radius: 6px; padding: 0 14px;")
            else:
                btn.setStyleSheet("border-radius: 6px; padding: 0 14px;")

        style_btn(self.btn_split_train, active == "train", "#22c55e")
        style_btn(self.btn_split_val, active in ("val", "validation"), "#f59e0b")
        style_btn(self.btn_split_test, active == "test", "#38bdf8")

        self.btn_manage_templates.setText(f"⚙️ Kelola Template ({active.capitalize()})")
        self.summary_badge.setText(f"Partisi: {active.upper()} ({len(self.images_list)} Frame)")

    def _switch_split(self, split_name: str):
        if self.current_image_name:
            self._save_current_prompts()
        self.active_split = (split_name or "train").lower().strip()
        self.current_image_name = ""
        self.reload_data()

    def _open_template_manager(self):
        tpl_path = self.dataset_manager.get_template_path_for_split(self.active_split)
        dlg = TemplateManagerDialog(
            template_file_path=tpl_path,
            split_name=self.active_split,
            parent=self
        )
        dlg.exec()

    def reload_data(self):
        """Reload image list for active split and refresh views."""
        self.images_list = self.dataset_manager.get_images_for_split(self.active_split)
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
        self._update_split_buttons_ui()

        if self.image_list_widget.count() > 0:
            target_idx = first_annotated_idx
            if self.current_image_name and self.current_image_name in self.images_list:
                target_idx = self.images_list.index(self.current_image_name)
            self.image_list_widget.setCurrentRow(target_idx)
        else:
            self.current_image_name = ""
            self.editor_title.setText("Daftar Variasi Prompt")
            self.preview_title.setText("🖼️ Preview Frame & Bounding Box")
            self.res_badge.setText("")
            self.image_preview_widget.set_frame("", [])
            self.objects_info_label.setText(f"Tidak ada frame di partisi '{self.active_split.upper()}'.<br>Atur pembagian frame pada Menu 3 (Split Dataset).")
            self._clear_prompt_rows()
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

    def _clear_all_current_prompts(self):
        """Delete all prompt variations for the currently selected frame with confirmation."""
        if not self.current_image_name:
            return

        if not self.prompt_rows:
            QMessageBox.information(self, "Info", "Tidak ada prompt yang tersimpan pada frame ini.")
            return

        count = len(self.prompt_rows)
        reply = QMessageBox.question(
            self,
            "Konfirmasi Hapus Semua Prompt",
            f"Apakah Anda yakin ingin menghapus seluruh ({count}) variasi prompt pada frame '{self.current_image_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self._clear_prompt_rows()
        self._save_current_prompts()
        self.dataset_updated.emit()

    def _on_duplicate_row(self, row_widget):
        data = row_widget.get_data()
        self._add_prompt_row_ui(data.get("user", ""), data.get("assistant", ""))
        self._save_current_prompts()

    def _update_list_item_for_image(self, img_name: str, new_prompt_count: Optional[int] = None):
        """Update display text and badge for a specific image item safely by matching Qt.UserRole."""
        if not img_name:
            return
        for idx in range(self.image_list_widget.count()):
            item = self.image_list_widget.item(idx)
            if item and item.data(Qt.UserRole) == img_name:
                anno = self.dataset_manager.get_annotation(img_name)
                b_count = len(anno.get("boxes", []))
                p_count = new_prompt_count if new_prompt_count is not None else len(anno.get("prompts", []))
                icon = "🟢" if b_count > 0 else "⚪"
                item.setText(f"{icon} [{b_count} box, {p_count} prompt] {img_name}")
                break

    def _save_current_prompts(self):
        if not self.current_image_name:
            return

        target_img = self.current_image_name
        anno = self.dataset_manager.get_annotation(target_img)
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
        self.dataset_manager.save_annotation(target_img, anno)

        # Update row item text in list safely by image identity
        self._update_list_item_for_image(target_img, new_prompt_count=len(new_prompts))

    # ---------------- Quick Generators ----------------
    def _on_execute_generator_clicked(self):
        """Execute selected prompt generation mode from dropdown."""
        mode = self.generator_mode_combo.currentData()
        if mode == "fast_template":
            self._open_fast_template_dialog()
        elif mode == "all_frames":
            self._auto_generate_all_frames()
        elif mode == "current_frame":
            self._auto_generate_from_boxes()
        elif mode == "frontier_all":
            self._add_negative_all_frames()
        elif mode == "frontier_current":
            self._add_negative_prompt()

    def _clear_all_dataset_prompts(self):
        """Delete all prompt variations across images in the active split with confirmation."""
        split_label = {"train": "Train", "val": "Validation", "test": "Test"}.get(self.active_split, self.active_split)
        images = self.images_list
        if not images:
            QMessageBox.information(self, "Split Kosong", f"Tidak ada gambar pada split '{split_label}'.")
            return

        total_prompts = 0
        frames_with_prompts = 0
        for img in images:
            anno = self.dataset_manager.get_annotation(img)
            p_len = len(anno.get("prompts", []))
            if p_len > 0:
                total_prompts += p_len
                frames_with_prompts += 1

        if total_prompts == 0:
            QMessageBox.information(self, "Info", f"Tidak ada prompt yang tersimpan pada split '{split_label}'.")
            return

        reply = QMessageBox.question(
            self,
            "Konfirmasi Hapus Semua Prompt",
            f"Apakah Anda yakin ingin menghapus seluruh <b>{total_prompts} prompt</b> pada <b>{frames_with_prompts} frame</b> di split '<b>{split_label}</b>'?\n\n"
            f"Tindakan ini hanya akan mengosongkan prompt pada split {split_label}.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        for img in images:
            anno = self.dataset_manager.get_annotation(img)
            if anno.get("prompts"):
                anno["prompts"] = []
                self.dataset_manager.save_annotation(img, anno)

        self.reload_data()
        self.dataset_updated.emit()
        QMessageBox.information(
            self,
            "Prompt Dibersihkan",
            f"Berhasil menghapus seluruh {total_prompts} prompt dari {frames_with_prompts} frame di split '{split_label}'."
        )

    def _open_fast_template_dialog(self):
        """Open Fast Prompt Generator dialog and execute template-based prompt creation for active split."""
        split_label = {"train": "Train", "val": "Validation", "test": "Test"}.get(self.active_split, self.active_split)
        images = self.images_list
        if not images:
            QMessageBox.information(
                self,
                f"Split {split_label} Kosong",
                f"Tidak ada gambar pada split '{split_label}'.\nSilakan atur pembagian frame terlebih dahulu di menu '3. Split Dataset'."
            )
            return

        initial_settings = getattr(self, "last_fast_generator_settings", None)
        if not initial_settings:
            initial_settings = self.dataset_manager.get_generator_settings()

        dialog = FastPromptGeneratorDialog(
            dataset_manager=self.dataset_manager,
            current_image_name=self.current_image_name,
            initial_settings=initial_settings,
            parent=self
        )
        if dialog.exec() != QDialog.Accepted:
            return

        settings = dialog.get_settings()
        self.last_fast_generator_settings = settings
        self.dataset_manager.save_generator_settings(settings)

        tpl_path = self.dataset_manager.get_template_path_for_split(self.active_split)
        gen = TemplatePromptGenerator(template_file_path=tpl_path, split=self.active_split)
        language = settings["language"]
        max_per_class = settings["max_per_class"]
        frontier_mode = settings.get("frontier_mode", "dynamic")
        positive_frontier_score = settings.get("positive_frontier_score", 0.85)
        frontier_score = settings["frontier_score"]
        scope = settings["scope"]
        clear_existing = settings["clear_existing"]
        project_classes = self.dataset_manager.classes

        target_images = self.images_list if scope == "all" else [self.current_image_name]
        total_frames_processed = 0
        total_prompts_created = 0

        for img_name in target_images:
            if not img_name:
                continue
            anno = self.dataset_manager.get_annotation(img_name)
            boxes = anno.get("boxes", [])

            existing = [] if clear_existing else list(anno.get("prompts", []))
            new_prompts = gen.generate_frame_prompts(
                boxes=boxes,
                project_classes=project_classes,
                language=language,
                max_templates_per_class=max_per_class,
                frontier_score=frontier_score,
                existing_prompts=existing,
                shuffle=True if max_per_class else False,
                class_synonyms=self.dataset_manager.get_all_synonyms(),
                frontier_mode=frontier_mode,
                positive_frontier_score=positive_frontier_score,
            )

            anno["prompts"] = existing + new_prompts
            self.dataset_manager.save_annotation(img_name, anno)
            total_frames_processed += 1
            total_prompts_created += len(new_prompts)

        if total_frames_processed == 0:
            QMessageBox.warning(
                self,
                "Tidak Ada Frame",
                f"Tidak ada frame yang dapat diproses pada split '{split_label}'."
            )
            return

        self.reload_data()
        self.dataset_updated.emit()
        if frontier_mode == "dynamic":
            mode_desc = "⚡ Dinamis (Ukuran BBOX)"
        elif frontier_mode == "static":
            mode_desc = f"🔒 Statis ({positive_frontier_score})"
        else:
            mode_desc = "🚫 Null / None (null)"
        QMessageBox.information(
            self,
            "Fast Generate Sukses! 🎉",
            f"Berhasil membuat {total_prompts_created} variasi prompt dari template split '{split_label}'!\n\n"
            f"• Split: {split_label}\n"
            f"• Template File: {os.path.basename(tpl_path)}\n"
            f"• Jumlah Frame Diproses: {total_frames_processed} frame\n"
            f"• Mode Frontier (Objek Positif): {mode_desc}\n"
            f"• Skor Frontier (Negatif/Koridor): {frontier_score}\n"
            f"• Format: Semantic Grounding JSON siap ekspor."
        )

    def _auto_generate_all_frames(self):
        """Generate prompt variations for ALL annotated frames in active split with Semantic Grounding JSON."""
        split_label = {"train": "Train", "val": "Validation", "test": "Test"}.get(self.active_split, self.active_split)
        images = self.images_list
        if not images:
            QMessageBox.information(self, "Info", f"Tidak ada gambar pada split '{split_label}'.")
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
                pos_score = TemplatePromptGenerator.calculate_dynamic_frontier_score(b_2d)
                payload = {
                    "target_detected": True,
                    "label": lbl,
                    "bounding_box": b_2d,
                    "frontier_score": pos_score,
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
                f"Tidak ditemukan frame dengan bounding box pada split '{split_label}'.\n"
                "Silakan kembali ke Menu 2 untuk menganotasi objek terlebih dahulu."
            )
            return

        self.reload_data()
        QMessageBox.information(
            self,
            "Batch Prompt Sukses! 🎉",
            f"Berhasil membuat {total_prompts_created} variasi prompt untuk {total_frames_processed} frame di split '{split_label}'!\n\n"
            "Format output: Semantic Grounding JSON siap latih."
        )

    def _add_negative_all_frames(self):
        """Add frontier exploration prompts with score for all annotated frames in active split."""
        split_label = {"train": "Train", "val": "Validation", "test": "Test"}.get(self.active_split, self.active_split)
        images = self.images_list
        if not images:
            QMessageBox.information(self, "Info", f"Tidak ada gambar pada split '{split_label}'.")
            return

        dlg = FrontierScoreDialog(
            current_score=0.85,
            title=f"Frontier Score untuk Frame Split {split_label}",
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
            f"Berhasil menambahkan {count} sampel frontier exploration (Score: {score}) ke frame split '{split_label}'."
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
            pos_score = TemplatePromptGenerator.calculate_dynamic_frontier_score(b_2d)
            payload = {
                "target_detected": True,
                "label": lbl,
                "bounding_box": b_2d,
                "frontier_score": pos_score,
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



