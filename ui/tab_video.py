import os
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QRadioButton, QButtonGroup, QSpinBox, QCheckBox,
    QLineEdit, QProgressBar, QMessageBox, QFrame, QGroupBox,
    QListWidget, QListWidgetItem, QAbstractItemView, QComboBox
)
from core.video_extractor import VideoExtractor


class VideoQueueItemWidget(QWidget):
    """Custom card row widget for each queued video with right-side split category dropdown."""
    role_changed = Signal(str, str)  # (video_path, new_role)

    def __init__(self, video_path: str, initial_role: str = "train_val", parent=None):
        super().__init__(parent)
        self.video_path = video_path
        file_name = os.path.basename(video_path)
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024) if os.path.isfile(video_path) else 0.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(10)

        self.name_label = QLabel(f"🎬 {file_name}")
        self.name_label.setStyleSheet("font-weight: 600; background: transparent;")
        self.size_label = QLabel(f"({file_size_mb:.1f} MB)")
        self.size_label.setStyleSheet("color: #94a3b8; font-size: 11px; background: transparent;")

        layout.addWidget(self.name_label)
        layout.addWidget(self.size_label)
        layout.addStretch()

        cat_lbl = QLabel("Kategori:")
        cat_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; background: transparent;")
        layout.addWidget(cat_lbl)

        self.role_combo = QComboBox()
        self.role_combo.addItem("🟢 Train & Val", "train_val")
        self.role_combo.addItem("🔵 Test Only", "test")
        self.role_combo.setFixedWidth(145)
        self.role_combo.setToolTip(
            "Pilih peran video ini untuk pembagian dataset di Menu 3:\n"
            "• Train & Val: Frame akan dibagi untuk pelatihan & validasi.\n"
            "• Test Only: 100% frame dari video ini khusus masuk ke Test Set (bebas data leakage)."
        )

        idx = 1 if (initial_role or "").lower().strip() == "test" else 0
        self.role_combo.setCurrentIndex(idx)
        self._update_combo_style()
        self.role_combo.currentIndexChanged.connect(self._on_combo_changed)
        layout.addWidget(self.role_combo)

    def get_role(self) -> str:
        return self.role_combo.currentData() or "train_val"

    def _on_combo_changed(self):
        self._update_combo_style()
        self.role_changed.emit(self.video_path, self.get_role())

    def _update_combo_style(self):
        role = self.get_role()
        if role == "test":
            self.role_combo.setStyleSheet(
                "QComboBox { border: 1px solid #38bdf8; color: #38bdf8; font-weight: 600; padding: 3px 8px; }"
            )
        else:
            self.role_combo.setStyleSheet(
                "QComboBox { border: 1px solid #22c55e; color: #22c55e; font-weight: 600; padding: 3px 8px; }"
            )


class VideoQueueListWidget(QListWidget):
    """Custom ListWidget that handles Delete key to remove items."""
    delete_pressed = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_pressed.emit()
        else:
            super().keyPressEvent(event)


class VideoExtractionThread(QThread):
    """Worker thread for extracting multiple video frames without freezing the GUI."""
    progress_updated = Signal(int, int, str, int, int, int)  # cur_v, tot_v, vid_name, cur_f, tot_f, total_saved
    finished_extraction = Signal(bool, str, int)

    def __init__(self, extractor, video_paths, output_dir, create_subfolders, mode, frame_interval, time_ms, max_size, prefix):
        super().__init__()
        self.extractor = extractor
        self.video_paths = video_paths
        self.output_dir = output_dir
        self.create_subfolders = create_subfolders
        self.mode = mode
        self.frame_interval = frame_interval
        self.time_ms = time_ms
        self.max_size = max_size
        self.prefix = prefix

    def run(self):
        def on_progress(cur_v, tot_v, v_name, cur_f, tot_f, saved):
            self.progress_updated.emit(cur_v, tot_v, v_name, cur_f, tot_f, saved)

        success, msg, saved = self.extractor.extract_batch(
            video_paths=self.video_paths,
            output_dir=self.output_dir,
            create_subfolders=self.create_subfolders,
            mode=self.mode,
            frame_interval=self.frame_interval,
            time_interval_ms=self.time_ms,
            max_size=self.max_size,
            prefix=self.prefix,
            progress_callback=on_progress,
        )
        self.finished_extraction.emit(success, msg, saved)


class TabVideo(QWidget):
    """Menu 1: Batch Video to Frames Extractor Tab."""
    frames_extracted = Signal(str)  # Emits output directory when done

    def __init__(self, dataset_manager, parent=None):
        super().__init__(parent)
        self.dataset_manager = dataset_manager
        self.extractor = VideoExtractor()
        self.worker_thread = None
        self._queued_video_roles: dict[str, str] = {}

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(14)

        # Settings Group Box
        settings_box = QGroupBox("Pengaturan Input Video & Folder Output")
        settings_layout = QVBoxLayout(settings_box)
        settings_layout.setSpacing(10)

        # Video List Header
        video_header = QHBoxLayout()
        video_label = QLabel("Antrean File Video:")
        video_label.setStyleSheet("font-weight: 700; color: #38bdf8;")
        self.video_count_label = QLabel("(0 video dipilih)")
        self.video_count_label.setStyleSheet("color: #94a3b8; font-size: 12px;")

        video_header.addWidget(video_label)
        video_header.addWidget(self.video_count_label)
        video_header.addStretch()

        self.add_videos_btn = QPushButton("➕ Tambah Video...")
        self.add_videos_btn.setObjectName("primaryBtn")
        self.add_videos_btn.setToolTip("Pilih satu atau lebih file video sekaligus (tahan Ctrl / Shift untuk memilih banyak file)")
        self.add_videos_btn.clicked.connect(self._browse_videos)

        self.remove_video_btn = QPushButton("🗑️ Hapus Terpilih")
        self.remove_video_btn.setEnabled(False)
        self.remove_video_btn.setToolTip("Hapus video yang dipilih dari antrean")
        self.remove_video_btn.clicked.connect(self._remove_selected_videos)

        self.clear_videos_btn = QPushButton("Hapus Semua")
        self.clear_videos_btn.setEnabled(False)
        self.clear_videos_btn.setToolTip("Kosongkan seluruh antrean video")
        self.clear_videos_btn.clicked.connect(self._clear_all_videos)

        video_header.addWidget(self.add_videos_btn)
        video_header.addWidget(self.remove_video_btn)
        video_header.addWidget(self.clear_videos_btn)
        settings_layout.addLayout(video_header)

        # Video Queue List
        self.video_list_widget = VideoQueueListWidget()
        self.video_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.video_list_widget.setFixedHeight(225)
        self.video_list_widget.setToolTip("Daftar video yang akan diproses. Pilih kategori (Train & Val / Test Only) di sisi kanan tiap item.")
        self.video_list_widget.delete_pressed.connect(self._remove_selected_videos)
        self.video_list_widget.itemSelectionChanged.connect(self._on_item_selection_changed)
        settings_layout.addWidget(self.video_list_widget)

        # Output Folder Row
        output_row = QHBoxLayout()
        output_label = QLabel("Folder Output:")
        output_label.setFixedWidth(130)
        self.output_dir_edit = QLineEdit()
        default_out = self.dataset_manager.frames_dir or ""
        self.output_dir_edit.setText(default_out)
        self.output_dir_edit.setPlaceholderText("Folder penyimpanan frame...")
        self.browse_output_btn = QPushButton("Pilih Folder...")
        self.browse_output_btn.clicked.connect(self._browse_output)
        output_row.addWidget(output_label)
        output_row.addWidget(self.output_dir_edit)
        output_row.addWidget(self.browse_output_btn)
        settings_layout.addLayout(output_row)

        # Target Folder Structure: Merge vs Subfolders
        struct_row = QHBoxLayout()
        struct_lbl = QLabel("Struktur Frame:")
        struct_lbl.setFixedWidth(130)
        struct_lbl.setStyleSheet("font-weight: 600; color: #cbd5e1;")

        self.radio_merge = QRadioButton("Gabung Semua Video ke 1 Folder (Merged)")
        self.radio_merge.setChecked(True)
        self.radio_merge.setToolTip(
            "Semua frame dari seluruh video digabung langsung ke folder output.\n"
            "Direkomendasikan agar di Menu 2 semua frame langsung muncul bersamaan dalam 1 daftar."
        )

        self.radio_subfolder = QRadioButton("Pisahkan ke Subfolder per Video")
        self.radio_subfolder.setToolTip(
            "Setiap video akan disimpan dalam subfolder terpisah (misal: frames/video_01/, frames/video_02/)."
        )

        self.struct_group = QButtonGroup(self)
        self.struct_group.addButton(self.radio_merge)
        self.struct_group.addButton(self.radio_subfolder)

        struct_row.addWidget(struct_lbl)
        struct_row.addWidget(self.radio_merge)
        struct_row.addWidget(self.radio_subfolder)
        struct_row.addStretch()
        settings_layout.addLayout(struct_row)

        # File Prefix Row
        prefix_row = QHBoxLayout()
        prefix_label = QLabel("Prefix Nama File:")
        prefix_label.setFixedWidth(130)
        self.prefix_edit = QLineEdit("amr_frame")
        self.prefix_edit.setFixedWidth(150)
        prefix_row.addWidget(prefix_label)
        prefix_row.addWidget(self.prefix_edit)
        prefix_row.addStretch()
        settings_layout.addLayout(prefix_row)

        main_layout.addWidget(settings_box)

        # Extraction Mode Box
        mode_box = QGroupBox("Mode & Interval Pengambilan Frame")
        mode_layout = QVBoxLayout(mode_box)
        mode_layout.setSpacing(14)

        # Mode A: By Frame
        mode_frame_row = QHBoxLayout()
        self.radio_frame = QRadioButton("Ambil tiap N frame:")
        self.radio_frame.setChecked(True)
        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(1, 1000)
        self.spin_frame.setValue(15)
        self.spin_frame.setSuffix(" frame")
        mode_frame_row.addWidget(self.radio_frame)
        mode_frame_row.addWidget(self.spin_frame)
        mode_frame_row.addStretch()
        mode_layout.addLayout(mode_frame_row)

        # Mode B: By Time
        mode_time_row = QHBoxLayout()
        self.radio_time = QRadioButton("Ambil tiap rentang waktu:")
        self.spin_time = QSpinBox()
        self.spin_time.setRange(50, 60000)
        self.spin_time.setValue(500)
        self.spin_time.setSingleStep(50)
        self.spin_time.setSuffix(" ms (milidetik)")
        mode_time_row.addWidget(self.radio_time)
        mode_time_row.addWidget(self.spin_time)
        mode_time_row.addStretch()
        mode_layout.addLayout(mode_time_row)

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.radio_frame)
        self.mode_group.addButton(self.radio_time)

        # Resize Option
        resize_row = QHBoxLayout()
        self.check_resize = QCheckBox("Batasi resolusi maksimum (Resize):")
        self.spin_resize = QSpinBox()
        self.spin_resize.setRange(256, 4096)
        self.spin_resize.setValue(1024)
        self.spin_resize.setSuffix(" px")
        self.spin_resize.setEnabled(False)
        self.check_resize.toggled.connect(self.spin_resize.setEnabled)
        resize_row.addWidget(self.check_resize)
        resize_row.addWidget(self.spin_resize)
        resize_row.addStretch()
        mode_layout.addLayout(resize_row)

        main_layout.addWidget(mode_box)

        # Progress & Actions Card
        action_card = QFrame()
        action_card.setObjectName("card")
        action_layout = QVBoxLayout(action_card)
        action_layout.setSpacing(12)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        action_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Status: Menunggu file video...")
        self.status_label.setObjectName("subtitleLabel")
        action_layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("⚡ Mulai Ekstraksi Frame")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.setFixedHeight(38)
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start_extraction)

        self.cancel_btn = QPushButton("Batal")
        self.cancel_btn.setObjectName("dangerBtn")
        self.cancel_btn.setFixedHeight(38)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_extraction)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.cancel_btn)
        action_layout.addLayout(btn_row)

        main_layout.addWidget(action_card)
        main_layout.addStretch()

    def update_output_dir(self, directory: str):
        """Update output folder when project directory changes."""
        self.output_dir_edit.setText(directory)

    def _browse_videos(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Pilih File Video (Bisa Pilih Banyak Sekaligus)",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;Semua File (*.*)",
        )
        if file_paths:
            existing = {
                self.video_list_widget.item(i).data(Qt.UserRole)
                for i in range(self.video_list_widget.count())
            }
            saved_roles = getattr(self.dataset_manager, "video_roles", {}) or {}
            added_count = 0
            for path in file_paths:
                if path not in existing:
                    base_name = os.path.basename(path)
                    initial_role = saved_roles.get(base_name, "train_val")

                    item = QListWidgetItem()
                    item.setData(Qt.UserRole, path)
                    item.setToolTip(path)
                    item.setSizeHint(QSize(0, 42))

                    row_widget = VideoQueueItemWidget(path, initial_role=initial_role)
                    row_widget.role_changed.connect(lambda *_: self._update_video_count())

                    self.video_list_widget.addItem(item)
                    self.video_list_widget.setItemWidget(item, row_widget)
                    existing.add(path)
                    added_count += 1

            self._update_video_count()
            if added_count > 0:
                self.status_label.setText(f"{self.video_list_widget.count()} video siap diekstrak.")

    def _remove_selected_videos(self):
        selected_items = self.video_list_widget.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            row = self.video_list_widget.row(item)
            self.video_list_widget.takeItem(row)
        self._update_video_count()

    def _clear_all_videos(self):
        if self.video_list_widget.count() == 0:
            return
        self.video_list_widget.clear()
        self._update_video_count()
        self.status_label.setText("Status: Menunggu file video...")

    def _on_item_selection_changed(self):
        selected = len(self.video_list_widget.selectedItems())
        self.remove_video_btn.setEnabled(selected > 0)

    def _update_video_count(self):
        cnt = self.video_list_widget.count()
        if cnt > 0:
            tv_cnt = 0
            test_cnt = 0
            for i in range(cnt):
                item = self.video_list_widget.item(i)
                w = self.video_list_widget.itemWidget(item)
                role = w.get_role() if w else "train_val"
                if role == "test":
                    test_cnt += 1
                else:
                    tv_cnt += 1
            self.video_count_label.setText(f"({cnt} video dipilih: {tv_cnt} Train & Val • {test_cnt} Test Only)")
        else:
            self.video_count_label.setText("(0 video dipilih)")

        self.clear_videos_btn.setEnabled(cnt > 0)
        self.start_btn.setEnabled(cnt > 0)
        self.remove_video_btn.setEnabled(len(self.video_list_widget.selectedItems()) > 0)
        if cnt == 0:
            self.status_label.setText("Status: Menunggu file video...")

    def _browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Pilih Folder Output Frame")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _start_extraction(self):
        video_paths = []
        self._queued_video_roles = {}
        for i in range(self.video_list_widget.count()):
            item = self.video_list_widget.item(i)
            path = item.data(Qt.UserRole)
            w = self.video_list_widget.itemWidget(item)
            role = w.get_role() if w else "train_val"
            video_paths.append(path)
            self._queued_video_roles[path] = role

        output_dir = self.output_dir_edit.text().strip()

        if not video_paths:
            QMessageBox.warning(self, "Peringatan", "Silakan tambahkan minimal 1 file video terlebih dahulu.")
            return

        if not output_dir:
            QMessageBox.warning(self, "Peringatan", "Silakan tentukan folder output penyimpanan frame.")
            return

        mode = "frame" if self.radio_frame.isChecked() else "time"
        frame_interval = self.spin_frame.value()
        time_ms = self.spin_time.value()
        max_size = self.spin_resize.value() if self.check_resize.isChecked() else None
        prefix = self.prefix_edit.text().strip() or "amr_frame"
        create_subfolders = self.radio_subfolder.isChecked()

        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.add_videos_btn.setEnabled(False)
        self.remove_video_btn.setEnabled(False)
        self.clear_videos_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Memulai proses ekstraksi untuk {len(video_paths)} video...")

        self.worker_thread = VideoExtractionThread(
            extractor=self.extractor,
            video_paths=video_paths,
            output_dir=output_dir,
            create_subfolders=create_subfolders,
            mode=mode,
            frame_interval=frame_interval,
            time_ms=time_ms,
            max_size=max_size,
            prefix=prefix,
        )
        self.worker_thread.progress_updated.connect(self._on_progress)
        self.worker_thread.finished_extraction.connect(self._on_finished)
        self.worker_thread.start()

    def _cancel_extraction(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.extractor.cancel()
            self.status_label.setText("Membatalkan proses...")
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, cur_v, tot_v, vid_name, cur_f, tot_f, total_saved):
        overall_pct = 0
        if tot_v > 0:
            frame_ratio = (cur_f / float(tot_f)) if tot_f > 0 else 0.0
            overall_pct = int(((cur_v - 1 + frame_ratio) / float(tot_v)) * 100)
            overall_pct = max(0, min(100, overall_pct))

        self.progress_bar.setValue(overall_pct)
        frame_pct = int((cur_f / float(tot_f) * 100)) if tot_f > 0 else 0
        self.status_label.setText(
            f"Memproses Video {cur_v}/{tot_v}: {vid_name} ({frame_pct}%) | "
            f"Frame: {cur_f}/{tot_f} | Total Tersimpan: {total_saved} frame"
        )

    def _on_finished(self, success, message, saved_count):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.add_videos_btn.setEnabled(True)
        self._update_video_count()

        # Register video & frame split roles in dataset_manager
        if hasattr(self.dataset_manager, "register_video_frames"):
            for v_path, role in self._queued_video_roles.items():
                extracted_files = self.extractor.last_batch_results.get(v_path, [])
                self.dataset_manager.register_video_frames(
                    os.path.basename(v_path), role, extracted_files
                )

        if success:
            self.progress_bar.setValue(100)
            self.status_label.setText(f"Selesai! {message}")
            QMessageBox.information(
                self,
                "Ekstraksi Selesai! 🎉",
                f"{message}\n\nKategori video (Train & Val / Test Only) telah disimpan.\n"
                f"Silakan lanjut ke Menu 2 untuk kurasi dan anotasi objek."
            )
            self.frames_extracted.emit(self.output_dir_edit.text().strip())
        else:
            self.status_label.setText(f"Ekstraksi berhenti: {message}")
            QMessageBox.warning(self, "Info Ekstraksi", message)

