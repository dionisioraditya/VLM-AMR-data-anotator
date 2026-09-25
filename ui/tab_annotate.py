import os
import time
from PySide6.QtCore import Qt, Signal, QThread, QEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QComboBox, QLineEdit, QSplitter,
    QMessageBox, QFrame, QInputDialog, QToolButton, QDialog,
    QProgressBar, QCheckBox, QFileDialog, QAbstractItemView, QDoubleSpinBox
)
from ui.components.canvas import AnnotationCanvas
from core.gemini_client import GeminiClient

class GeminiDetectionThread(QThread):
    """Worker thread to run Gemini Vision Grounding for a single image."""
    detection_finished = Signal(bool, str, list)

    def __init__(self, gemini_client, image_path, target_classes, custom_instructions):
        super().__init__()
        self.gemini_client = gemini_client
        self.image_path = image_path
        self.target_classes = target_classes
        self.custom_instructions = custom_instructions

    def run(self):
        success, msg, boxes = self.gemini_client.detect_objects(
            image_path=self.image_path,
            target_classes=self.target_classes,
            custom_instructions=self.custom_instructions,
        )
        self.detection_finished.emit(success, msg, boxes)


class GeminiBatchWorker(QThread):
    """Worker thread for batch autolabeling multiple images with Gemini and rate limit handling."""
    progress = Signal(int, int, str, str)  # current, total, image_name, status_message
    finished_batch = Signal(int, int, str)  # processed_count, detected_boxes, summary_message

    def __init__(self, gemini_client, dataset_manager, image_names, only_unannotated, target_classes, delay_seconds=4.0):
        super().__init__()
        self.gemini_client = gemini_client
        self.dataset_manager = dataset_manager
        self.image_names = list(image_names)
        self.only_unannotated = only_unannotated
        self.target_classes = target_classes
        self.delay_seconds = float(delay_seconds)
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        total = len(self.image_names)
        processed = 0
        detected_boxes = 0

        for idx, img_name in enumerate(self.image_names):
            if self._is_cancelled:
                self.finished_batch.emit(processed, detected_boxes, "Proses batch dihentikan oleh pengguna.")
                return

            if self.only_unannotated and self.dataset_manager.is_annotated(img_name):
                self.progress.emit(idx + 1, total, img_name, f"Melewati {img_name} (sudah dianotasi)")
                continue

            img_path = self.dataset_manager.get_image_path(img_name)
            self.progress.emit(idx + 1, total, img_name, f"Mendeteksi {img_name} via Gemini...")

            # Retry loop with exponential backoff on HTTP 429 (Rate Limit)
            max_retries = 3
            success = False
            msg = ""
            boxes = []

            for attempt in range(max_retries):
                if self._is_cancelled:
                    break
                success, msg, boxes = self.gemini_client.detect_objects(
                    image_path=img_path,
                    target_classes=self.target_classes,
                )
                if success:
                    break
                elif "429" in msg or "quota" in msg.lower() or "rate" in msg.lower() or "exhausted" in msg.lower():
                    wait_sec = 6 * (attempt + 1)
                    self.progress.emit(idx + 1, total, img_name, f"⏳ Rate limit tercapai, jeda {wait_sec}s lalu coba lagi ({attempt+1}/{max_retries})...")
                    time.sleep(wait_sec)
                else:
                    # Non-retryable error
                    break

            if success:
                anno = self.dataset_manager.get_annotation(img_name)
                anno["boxes"] = boxes
                self.dataset_manager.save_annotation(img_name, anno)
                processed += 1
                detected_boxes += len(boxes)
                status = f"✅ {img_name}: {len(boxes)} objek terdeteksi"
            else:
                status = f"⚠️ {img_name}: {msg}"

            self.progress.emit(idx + 1, total, img_name, status)

            # Polite delay between requests to stay within 15 RPM
            if self.delay_seconds > 0 and idx < total - 1 and not self._is_cancelled:
                time.sleep(self.delay_seconds)

        self.finished_batch.emit(
            processed,
            detected_boxes,
            f"Selesai! {processed} frame berhasil diproses ({detected_boxes} total objek terdeteksi)."
        )


class BatchDetectionDialog(QDialog):
    """Dialog for running Batch Auto-Detect with progress bar, model selection, and rate limiting."""

    def __init__(self, tab_annotate, parent=None):
        super().__init__(parent)
        self.tab_annotate = tab_annotate
        self.dataset_manager = tab_annotate.dataset_manager
        self.gemini_client = tab_annotate.gemini_client
        self.worker = None

        self.setWindowTitle("⚡ Batch Auto-Detect (Gemini AI)")
        self.setFixedWidth(540)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QLabel("⚡ Batch Auto-Grounding Bounding Box")
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        folder_info = QLabel(f"Folder aktif: <b>{os.path.basename(self.dataset_manager.active_frames_dir or '')}</b>")
        layout.addWidget(folder_info)

        # Model Selection with quota information
        model_row = QHBoxLayout()
        model_lbl = QLabel("Pilih Model Gemini:")
        model_lbl.setFixedWidth(140)
        self.model_combo = QComboBox()
        self.model_combo.addItem("gemini-2.0-flash (Rekomendasi: 1,500 req/hari)", "gemini-2.0-flash")
        self.model_combo.addItem("gemini-1.5-flash (1,500 req/hari)", "gemini-1.5-flash")
        self.model_combo.addItem("gemini-2.5-flash (Preview: Limit ketat 20 req/hari)", "gemini-2.5-flash")
        model_row.addWidget(model_lbl)
        model_row.addWidget(self.model_combo)
        layout.addLayout(model_row)

        # Delay SpinBox to avoid 15 RPM rate limits
        delay_row = QHBoxLayout()
        delay_lbl = QLabel("Jeda Antar Gambar:")
        delay_lbl.setFixedWidth(140)
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.5, 30.0)
        self.delay_spin.setValue(4.0)
        self.delay_spin.setSingleStep(0.5)
        self.delay_spin.setSuffix(" detik (Rekomendasi Free Tier 15 RPM)")
        delay_row.addWidget(delay_lbl)
        delay_row.addWidget(self.delay_spin)
        layout.addLayout(delay_row)

        self.only_unannotated_check = QCheckBox("Hanya proses frame yang belum dianotasi (⚪)")
        self.only_unannotated_check.setChecked(True)
        layout.addWidget(self.only_unannotated_check)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(20)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Klik 'Mulai Batch' untuk memulai deteksi otomatis.")
        self.status_label.setObjectName("subtitleLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.start_btn = QPushButton("▶ Mulai Batch")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.clicked.connect(self._start_batch)
        btn_row.addWidget(self.start_btn)

        self.close_btn = QPushButton("Tutup")
        self.close_btn.clicked.connect(self._on_close)
        btn_row.addWidget(self.close_btn)

        layout.addLayout(btn_row)

    def _start_batch(self):
        if not self.gemini_client.api_key:
            QMessageBox.warning(self, "API Key Kosong", "Masukkan API Key Gemini di menu pengaturan terlebih dahulu.")
            return

        image_names = self.dataset_manager.get_image_list()
        if not image_names:
            QMessageBox.warning(self, "Folder Kosong", "Tidak ada gambar di folder aktif.")
            return

        # Set selected model
        selected_model = self.model_combo.currentData()
        self.gemini_client.set_model(selected_model)

        self.start_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.delay_spin.setEnabled(False)
        self.only_unannotated_check.setEnabled(False)
        self.close_btn.setText("Hentikan")
        self.progress_bar.setValue(0)

        self.worker = GeminiBatchWorker(
            gemini_client=self.gemini_client,
            dataset_manager=self.dataset_manager,
            image_names=image_names,
            only_unannotated=self.only_unannotated_check.isChecked(),
            target_classes=self.dataset_manager.classes,
            delay_seconds=self.delay_spin.value(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_batch.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, current, total, img_name, status):
        pct = int((current / float(total)) * 100)
        self.progress_bar.setValue(pct)
        self.status_label.setText(f"[{current}/{total}] {status}")

    def _on_finished(self, processed, boxes, message):
        self.start_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.delay_spin.setEnabled(True)
        self.only_unannotated_check.setEnabled(True)
        self.close_btn.setText("Selesai")
        self.status_label.setText(message)
        self.tab_annotate.reload_images()
        self.tab_annotate.dataset_updated.emit()
        QMessageBox.information(self, "Batch Selesai", message)

    def _on_close(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.status_label.setText("Menghentikan proses...")
            self.close_btn.setEnabled(False)
        else:
            self.accept()

class ClassManagerDialog(QDialog):
    """Dialog to view, add, and remove target class labels in bulk."""

    def __init__(self, dataset_manager, parent=None):
        super().__init__(parent)
        self.dataset_manager = dataset_manager
        self.setWindowTitle("Kelola Daftar Label")
        self.setFixedWidth(380)
        self.setFixedHeight(440)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        title = QLabel("⚙️ Kelola Daftar Label")
        title.setObjectName("titleLabel")
        layout.addWidget(title)

        subtitle = QLabel("Pilih label yang ingin dihapus, atau tambahkan label baru:")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # List of classes
        self.list_widget = QListWidget()
        for c in self.dataset_manager.classes:
            self.list_widget.addItem(c)
        layout.addWidget(self.list_widget)

        # Add input row
        add_row = QHBoxLayout()
        self.new_class_edit = QLineEdit()
        self.new_class_edit.setPlaceholderText("Nama label baru (misal: dispenser)...")
        add_btn = QPushButton("➕ Tambah")
        add_btn.setObjectName("primaryBtn")
        add_btn.clicked.connect(self._add_class)
        add_row.addWidget(self.new_class_edit)
        add_row.addWidget(add_btn)
        layout.addLayout(add_row)

        # Delete selected button
        del_btn = QPushButton("🗑️ Hapus Label Terpilih")
        del_btn.setObjectName("dangerBtn")
        del_btn.clicked.connect(self._delete_selected)
        layout.addWidget(del_btn)

        # Close button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Selesai")
        close_btn.setObjectName("primaryBtn")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _add_class(self):
        txt = self.new_class_edit.text().strip().lower()
        if txt and txt not in self.dataset_manager.classes:
            self.dataset_manager.add_class(txt)
            self.list_widget.addItem(txt)
            self.new_class_edit.clear()

    def _delete_selected(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            item = self.list_widget.takeItem(row)
            lbl = item.text()
            self.dataset_manager.remove_class(lbl)


class TabAnnotate(QWidget):
    """Menu 2: Frame Curation and AI-Assisted Bounding Box Annotation."""
    dataset_updated = Signal()

    def __init__(self, dataset_manager, gemini_client, parent=None):
        super().__init__(parent)
        self.dataset_manager = dataset_manager
        self.gemini_client = gemini_client
        self.ai_thread = None

        self.current_image_name = ""
        self.images_list = []

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # ---------------- Top Bar: Folder Selector & Stats ----------------
        top_bar = QFrame()
        top_bar.setObjectName("card")
        top_bar.setFixedHeight(54)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(14, 6, 14, 6)
        top_layout.setSpacing(10)

        self.info_label = QLabel("🏷️ Anotasi & Kurasi")
        self.info_label.setObjectName("titleLabel")
        top_layout.addWidget(self.info_label)

        # Folder Selector
        folder_lbl = QLabel("Folder Frame:")
        folder_lbl.setStyleSheet("font-weight: 600; color: #94a3b8;")
        top_layout.addWidget(folder_lbl)

        self.folder_combo = QComboBox()
        self.folder_combo.setMinimumWidth(220)
        self.folder_combo.currentIndexChanged.connect(self._on_folder_combo_changed)
        top_layout.addWidget(self.folder_combo)

        self.browse_folder_btn = QPushButton("📁 Ganti...")
        self.browse_folder_btn.clicked.connect(self._browse_custom_folder)
        top_layout.addWidget(self.browse_folder_btn)

        top_layout.addStretch()

        self.stats_badge = QLabel("0 Frame | 0 Teranotasi")
        self.stats_badge.setObjectName("badgeLabel")
        top_layout.addWidget(self.stats_badge)

        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.reload_images)
        top_layout.addWidget(self.refresh_btn)

        main_layout.addWidget(top_bar)

        # Main Content Splitter (Left: File List, Center: Canvas, Right: Tools & Boxes)
        splitter = QSplitter(Qt.Horizontal)

        # ---------------- Left Panel: Image Gallery & Navigation ----------------
        left_panel = QFrame()
        left_panel.setObjectName("panelCard")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        left_title = QLabel("Daftar Frame")
        left_title.setStyleSheet("font-weight: 700; color: #38bdf8;")
        left_layout.addWidget(left_title)

        # Search filter
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Cari frame...")
        self.search_edit.textChanged.connect(self._filter_images)
        left_layout.addWidget(self.search_edit)

        # List Widget (supports Shift/Ctrl multi-selection)
        self.image_list_widget = QListWidget()
        self.image_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.image_list_widget.currentRowChanged.connect(self._on_image_selected)
        self.image_list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.image_list_widget.installEventFilter(self)
        left_layout.addWidget(self.image_list_widget)

        # Delete & Navigation Controls
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev (A)")
        self.prev_btn.clicked.connect(self._prev_image)
        self.next_btn = QPushButton("Next (D) ▶")
        self.next_btn.clicked.connect(self._next_image)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)
        left_layout.addLayout(nav_layout)

        self.delete_frame_btn = QPushButton("🗑️ Hapus Frame Ini (Del)")
        self.delete_frame_btn.setObjectName("dangerBtn")
        self.delete_frame_btn.clicked.connect(self._delete_current_frame)
        left_layout.addWidget(self.delete_frame_btn)

        left_panel.setMinimumWidth(230)
        left_panel.setMaximumWidth(290)
        splitter.addWidget(left_panel)

        # ---------------- Center Panel: Canvas & View Controls ----------------
        center_panel = QFrame()
        center_panel.setObjectName("panelCard")
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(6)

        # Canvas
        self.canvas = AnnotationCanvas(self)
        self.canvas.boxes_changed.connect(self._on_boxes_changed)
        self.canvas.box_selected.connect(self._on_canvas_box_selected)
        center_layout.addWidget(self.canvas)

        # Canvas View Controls Bar
        canvas_bar = QHBoxLayout()
        self.canvas_status = QLabel("Klik & drag untuk gambar box. Spasi + Drag untuk Pan.")
        self.canvas_status.setObjectName("subtitleLabel")
        canvas_bar.addWidget(self.canvas_status)

        canvas_bar.addStretch()

        fit_btn = QToolButton()
        fit_btn.setText("Fit")
        fit_btn.clicked.connect(lambda: self.canvas.fitInView(self.canvas.sceneRect(), Qt.KeepAspectRatio))
        canvas_bar.addWidget(fit_btn)

        zoom_in_btn = QToolButton()
        zoom_in_btn.setText("➕")
        zoom_in_btn.clicked.connect(lambda: self.canvas.scale(1.2, 1.2))
        canvas_bar.addWidget(zoom_in_btn)

        zoom_out_btn = QToolButton()
        zoom_out_btn.setText("➖")
        zoom_out_btn.clicked.connect(lambda: self.canvas.scale(1.0 / 1.2, 1.0 / 1.2))
        canvas_bar.addWidget(zoom_out_btn)

        center_layout.addLayout(canvas_bar)
        splitter.addWidget(center_panel)

        # ---------------- Right Panel: Tools, AI Auto-Detect, Boxes List ----------------
        right_panel = QFrame()
        right_panel.setObjectName("panelCard")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)

        # Active Class / Label
        class_header = QLabel("Label Aktif:")
        class_header.setStyleSheet("font-weight: 700; color: #38bdf8;")
        right_layout.addWidget(class_header)

        class_row = QHBoxLayout()
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.dataset_manager.classes)
        self.class_combo.currentTextChanged.connect(self._on_class_changed)
        class_row.addWidget(self.class_combo)

        add_class_btn = QPushButton("➕")
        add_class_btn.setFixedWidth(36)
        add_class_btn.setToolTip("Tambah label baru")
        add_class_btn.clicked.connect(self._add_new_class)
        class_row.addWidget(add_class_btn)

        del_class_btn = QPushButton("🗑️")
        del_class_btn.setObjectName("dangerBtn")
        del_class_btn.setFixedWidth(36)
        del_class_btn.setToolTip("Hapus label yang sedang dipilih")
        del_class_btn.clicked.connect(self._delete_current_class)
        class_row.addWidget(del_class_btn)

        manage_class_btn = QPushButton("⚙️ Kelola")
        manage_class_btn.setToolTip("Lihat dan kelola seluruh daftar label")
        manage_class_btn.clicked.connect(self._open_class_manager)
        class_row.addWidget(manage_class_btn)

        right_layout.addLayout(class_row)

        # AI Grounding Section
        ai_card = QFrame()
        ai_card.setObjectName("card")
        ai_layout = QVBoxLayout(ai_card)
        ai_layout.setContentsMargins(10, 10, 10, 10)
        ai_layout.setSpacing(8)

        ai_title = QLabel("⚡ Gemini AI Auto-Grounding")
        ai_title.setStyleSheet("font-weight: 700; color: #a855f7;")
        ai_layout.addWidget(ai_title)

        self.model_combo = QComboBox()
        self.model_combo.addItems(GeminiClient.DEFAULT_MODELS)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        ai_layout.addWidget(self.model_combo)

        self.ai_detect_btn = QPushButton("✨ Auto-Detect Frame Ini")
        self.ai_detect_btn.setObjectName("aiBtn")
        self.ai_detect_btn.setFixedHeight(34)
        self.ai_detect_btn.clicked.connect(self._run_ai_detection)
        ai_layout.addWidget(self.ai_detect_btn)

        self.batch_ai_btn = QPushButton("⚡ Batch Auto-Detect Semua Frame")
        self.batch_ai_btn.setObjectName("primaryBtn")
        self.batch_ai_btn.setFixedHeight(34)
        self.batch_ai_btn.clicked.connect(self._open_batch_dialog)
        ai_layout.addWidget(self.batch_ai_btn)

        self.ai_status_label = QLabel("Status: Siap")
        self.ai_status_label.setObjectName("subtitleLabel")
        ai_layout.addWidget(self.ai_status_label)

        right_layout.addWidget(ai_card)

        # Box List in Current Image
        boxes_header = QLabel("Objek Terdeteksi di Frame Ini:")
        boxes_header.setStyleSheet("font-weight: 700; color: #38bdf8;")
        right_layout.addWidget(boxes_header)

        self.box_list_widget = QListWidget()
        self.box_list_widget.currentRowChanged.connect(self._on_box_item_clicked)
        right_layout.addWidget(self.box_list_widget)

        # Box Action Buttons
        box_btn_row = QHBoxLayout()
        self.del_box_btn = QPushButton("Hapus Box (Del)")
        self.del_box_btn.clicked.connect(self.canvas.delete_selected_box)

        self.clear_boxes_btn = QPushButton("Hapus Semua")
        self.clear_boxes_btn.setObjectName("dangerBtn")
        self.clear_boxes_btn.clicked.connect(self.canvas.clear_all_boxes)

        box_btn_row.addWidget(self.del_box_btn)
        box_btn_row.addWidget(self.clear_boxes_btn)
        right_layout.addLayout(box_btn_row)

        right_panel.setMinimumWidth(280)
        right_panel.setMaximumWidth(340)
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        main_layout.addWidget(splitter)

    # ---------------- Folder Management ----------------
    def _update_folder_combo(self):
        """Update the subfolder dropdown with available folders."""
        self.folder_combo.blockSignals(True)
        self.folder_combo.clear()

        subfolders = self.dataset_manager.get_subfolders()
        active = os.path.abspath(self.dataset_manager.active_frames_dir or "")

        selected_idx = 0
        for idx, sub in enumerate(subfolders):
            sub_path = os.path.abspath(sub["path"])
            label = f"📁 {sub['name']} ({sub['count']} frame)"
            self.folder_combo.addItem(label, sub_path)
            if sub_path == active:
                selected_idx = idx

        self.folder_combo.setCurrentIndex(selected_idx)
        self.folder_combo.blockSignals(False)

    def _on_folder_combo_changed(self, index: int):
        if index < 0:
            return
        selected_path = self.folder_combo.currentData()
        if selected_path and os.path.isdir(selected_path):
            self.dataset_manager.set_active_frames_dir(selected_path)
            self.reload_images()

    def _browse_custom_folder(self):
        start_dir = self.dataset_manager.active_frames_dir or self.dataset_manager.frames_dir or ""
        chosen = QFileDialog.getExistingDirectory(self, "Pilih Folder Frame Gambar", start_dir)
        if chosen:
            self.dataset_manager.set_active_frames_dir(chosen)
            self._update_folder_combo()
            self.reload_images()

    # ---------------- Image List & Selection ----------------
    def reload_images(self, select_row: int = -1):
        """Reload image list from dataset manager."""
        self._update_folder_combo()
        self.images_list = self.dataset_manager.get_image_list()
        self._filter_images(self.search_edit.text(), select_row=select_row)
        self._update_stats()

    def _filter_images(self, query="", select_row: int = -1):
        q = query.strip().lower()
        self.image_list_widget.blockSignals(True)
        self.image_list_widget.clear()

        for img in self.images_list:
            if not q or q in img.lower():
                is_ann = self.dataset_manager.is_annotated(img)
                icon_prefix = "🟢" if is_ann else "⚪"
                item = QListWidgetItem(f"{icon_prefix} {img}")
                item.setData(Qt.UserRole, img)
                self.image_list_widget.addItem(item)

        self.image_list_widget.blockSignals(False)

        count = self.image_list_widget.count()
        if count > 0:
            target = max(0, min(select_row, count - 1)) if select_row >= 0 else 0
            self.image_list_widget.setCurrentRow(target)
            target_item = self.image_list_widget.item(target)
            if target_item:
                self.image_list_widget.scrollToItem(target_item)
        else:
            self.canvas.scene.clear()
            self.current_image_name = ""

    def _update_stats(self):
        total = len(self.images_list)
        annotated = sum(1 for img in self.images_list if self.dataset_manager.is_annotated(img))
        self.stats_badge.setText(f"{total} Frame | {annotated} Teranotasi")

    def _on_image_selected(self, row: int):
        if row < 0:
            return
        item = self.image_list_widget.item(row)
        if not item:
            return
        img_name = item.data(Qt.UserRole)
        self.load_image_by_name(img_name)

    def load_image_by_name(self, img_name: str):
        """Load an image and its annotations into the view."""
        self.current_image_name = img_name
        img_path = self.dataset_manager.get_image_path(img_name)
        if not os.path.isfile(img_path):
            return

        anno = self.dataset_manager.get_annotation(img_name)
        boxes = anno.get("boxes", [])

        self.canvas.load_image(img_path, boxes)
        self._refresh_box_list_ui()
        self.canvas_status.setText(f"Frame: {img_name} ({len(boxes)} objek teranotasi)")

    def _prev_image(self):
        curr = self.image_list_widget.currentRow()
        if curr > 0:
            self.image_list_widget.setCurrentRow(curr - 1)

    def _next_image(self):
        curr = self.image_list_widget.currentRow()
        if curr < self.image_list_widget.count() - 1:
            self.image_list_widget.setCurrentRow(curr + 1)

    def _on_selection_changed(self):
        selected_items = self.image_list_widget.selectedItems()
        count = len(selected_items)
        if count > 1:
            self.delete_frame_btn.setText(f"🗑️ Hapus {count} Frame (Del)")
            self.canvas_status.setText(f"{count} frame dipilih. Tekan Del untuk hapus semua frame yang dipilih.")
        else:
            self.delete_frame_btn.setText("🗑️ Hapus Frame Ini (Del)")

    def _delete_current_frame(self):
        selected_items = self.image_list_widget.selectedItems()
        if not selected_items:
            if not self.current_image_name:
                return
            selected_items = [self.image_list_widget.currentItem()]

        # Multiple deletion
        if len(selected_items) > 1:
            count = len(selected_items)
            img_names = [item.data(Qt.UserRole) for item in selected_items if item]
            rows = [self.image_list_widget.row(item) for item in selected_items if item]
            min_row = min(rows) if rows else 0

            msg_box = QMessageBox(
                QMessageBox.Question,
                "Konfirmasi Hapus Banyak Frame",
                f"Apakah Anda yakin ingin menghapus {count} frame yang dipilih sekaligus?",
                QMessageBox.Yes | QMessageBox.No,
                self,
            )
            msg_box.setDefaultButton(QMessageBox.Yes)
            reply = msg_box.exec()

            if reply == QMessageBox.Yes:
                for name in img_names:
                    if name:
                        self.dataset_manager.delete_image(name)
                # Next frame automatically shifts to min_row
                self.reload_images(select_row=min_row)
                self.dataset_updated.emit()
                self.image_list_widget.setFocus()
            return

        # Single deletion
        if not self.current_image_name:
            return

        current_row = self.image_list_widget.currentRow()

        msg_box = QMessageBox(
            QMessageBox.Question,
            "Konfirmasi Hapus Frame",
            f"Hapus frame '{self.current_image_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            self,
        )
        msg_box.setDefaultButton(QMessageBox.Yes)
        reply = msg_box.exec()

        if reply == QMessageBox.Yes:
            success = self.dataset_manager.delete_image(self.current_image_name)
            if success:
                # Next frame automatically shifts to current_row
                self.reload_images(select_row=current_row)
                self.dataset_updated.emit()
                self.image_list_widget.setFocus()

    # ---------------- Box Management & Sync ----------------
    def _on_boxes_changed(self):
        """Called whenever boxes are drawn, moved, or deleted on canvas."""
        if not self.current_image_name:
            return

        # Save to annotation
        anno = self.dataset_manager.get_annotation(self.current_image_name)
        anno["boxes"] = list(self.canvas.boxes)
        self.dataset_manager.save_annotation(self.current_image_name, anno)

        self._refresh_box_list_ui()
        self._update_current_list_item_icon()
        self._update_stats()
        self.dataset_updated.emit()

    def _update_current_list_item_icon(self):
        row = self.image_list_widget.currentRow()
        if row >= 0:
            item = self.image_list_widget.item(row)
            is_ann = len(self.canvas.boxes) > 0
            icon_prefix = "🟢" if is_ann else "⚪"
            item.setText(f"{icon_prefix} {self.current_image_name}")

    def _refresh_box_list_ui(self):
        self.box_list_widget.blockSignals(True)
        self.box_list_widget.clear()

        for idx, b in enumerate(self.canvas.boxes):
            lbl = b.get("label", "object")
            b_2d = b.get("box_2d", [0, 0, 0, 0])
            item = QListWidgetItem(f"[{lbl}] {b_2d}")
            self.box_list_widget.addItem(item)

        if 0 <= self.canvas.selected_index < self.box_list_widget.count():
            self.box_list_widget.setCurrentRow(self.canvas.selected_index)

        self.box_list_widget.blockSignals(False)

    def _on_box_item_clicked(self, row: int):
        self.canvas.selected_index = row
        self.canvas.viewport().update()

    def _on_canvas_box_selected(self, index: int):
        self.box_list_widget.blockSignals(True)
        self.box_list_widget.setCurrentRow(index)
        self.box_list_widget.blockSignals(False)

    def _on_class_changed(self, text: str):
        self.canvas.set_current_label(text)

    def _add_new_class(self):
        name, ok = QInputDialog.getText(self, "Tambah Label Baru", "Nama label (misal: pallet, forklift, cone):")
        if ok and name.strip():
            clean_name = name.strip().lower()
            self.dataset_manager.add_class(clean_name)
            if self.class_combo.findText(clean_name) == -1:
                self.class_combo.addItem(clean_name)
            self.class_combo.setCurrentText(clean_name)

    def _delete_current_class(self):
        curr_label = self.class_combo.currentText().strip()
        if not curr_label:
            return

        reply = QMessageBox.question(
            self,
            "Hapus Label",
            f"Hapus label '{curr_label}' dari daftar preset?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Yes:
            self.dataset_manager.remove_class(curr_label)
            idx = self.class_combo.findText(curr_label)
            if idx >= 0:
                self.class_combo.removeItem(idx)
            if self.class_combo.count() > 0:
                self.class_combo.setCurrentIndex(0)
            else:
                self.canvas.set_current_label("object")

    def _open_class_manager(self):
        dialog = ClassManagerDialog(self.dataset_manager, parent=self)
        if dialog.exec():
            # Refresh class_combo with updated classes
            curr = self.class_combo.currentText()
            self.class_combo.blockSignals(True)
            self.class_combo.clear()
            self.class_combo.addItems(self.dataset_manager.classes)
            idx = self.class_combo.findText(curr)
            if idx >= 0:
                self.class_combo.setCurrentIndex(idx)
            elif self.class_combo.count() > 0:
                self.class_combo.setCurrentIndex(0)
            self.class_combo.blockSignals(False)
            self.canvas.set_current_label(self.class_combo.currentText() or "object")

    def _on_model_changed(self, model_name: str):
        self.gemini_client.set_model(model_name)

    # ---------------- AI Auto Detection ----------------
    def _run_ai_detection(self):
        if not self.current_image_name:
            QMessageBox.warning(self, "Peringatan", "Pilih frame gambar terlebih dahulu.")
            return

        if not self.gemini_client.api_key:
            QMessageBox.warning(
                self,
                "API Key Belum Disetel",
                "Silakan masukkan Gemini API Key di menu pengaturan (pojok kanan atas).",
            )
            return

        img_path = self.dataset_manager.get_image_path(self.current_image_name)
        self.ai_detect_btn.setEnabled(False)
        self.ai_status_label.setText("Sedang mendeteksi via Gemini...")

        self.ai_thread = GeminiDetectionThread(
            gemini_client=self.gemini_client,
            image_path=img_path,
            target_classes=self.dataset_manager.classes,
            custom_instructions="Accurately detect AMR navigation objects and obstacles.",
        )
        self.ai_thread.detection_finished.connect(self._on_ai_detection_finished)
        self.ai_thread.start()

    def _on_ai_detection_finished(self, success: bool, message: str, boxes: list):
        self.ai_detect_btn.setEnabled(True)
        self.ai_status_label.setText(message)

        if success:
            merged = list(self.canvas.boxes) + boxes
            self.canvas.set_boxes(merged)
            self._on_boxes_changed()
            QMessageBox.information(self, "AI Grounding Sukses", f"{message}\nKotak deteksi telah ditambahkan ke canvas.")
        else:
            QMessageBox.warning(self, "AI Grounding Gagal", message)

    def _open_batch_dialog(self):
        dialog = BatchDetectionDialog(self, parent=self)
        dialog.exec()

    def keyPressEvent(self, event):
        # Keyboard shortcuts: A (prev), D (next), Del (delete)
        if event.key() == Qt.Key_A and not self.search_edit.hasFocus():
            self._prev_image()
        elif event.key() == Qt.Key_D and not self.search_edit.hasFocus():
            self._next_image()
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and not self.search_edit.hasFocus():
            if self.canvas.selected_index != -1:
                self.canvas.delete_selected_box()
            else:
                self._delete_current_frame()
        else:
            super().keyPressEvent(event)

    def eventFilter(self, source, event):
        if source == self.image_list_widget and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                self._delete_current_frame()
                return True
            elif event.key() == Qt.Key_A:
                self._prev_image()
                return True
            elif event.key() == Qt.Key_D:
                self._next_image()
                return True
        return super().eventFilter(source, event)
