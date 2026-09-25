import os
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QRadioButton, QButtonGroup, QSpinBox, QCheckBox,
    QLineEdit, QProgressBar, QMessageBox, QFrame, QGroupBox
)
from core.video_extractor import VideoExtractor

class VideoExtractionThread(QThread):
    """Worker thread for extracting video frames without freezing the GUI."""
    progress_updated = Signal(int, int, int)  # current, total, saved
    finished_extraction = Signal(bool, str, int)

    def __init__(self, extractor, video_path, output_dir, mode, frame_interval, time_ms, max_size, prefix):
        super().__init__()
        self.extractor = extractor
        self.video_path = video_path
        self.output_dir = output_dir
        self.mode = mode
        self.frame_interval = frame_interval
        self.time_ms = time_ms
        self.max_size = max_size
        self.prefix = prefix

    def run(self):
        def on_progress(cur, total, saved):
            self.progress_updated.emit(cur, total, saved)

        success, msg, saved = self.extractor.extract(
            video_path=self.video_path,
            output_dir=self.output_dir,
            mode=self.mode,
            frame_interval=self.frame_interval,
            time_interval_ms=self.time_ms,
            max_size=self.max_size,
            prefix=self.prefix,
            progress_callback=on_progress,
        )
        self.finished_extraction.emit(success, msg, saved)


class TabVideo(QWidget):
    """Menu 1: Video to Frames Extractor Tab."""
    frames_extracted = Signal(str)  # Emits output directory when done

    def __init__(self, dataset_manager, parent=None):
        super().__init__(parent)
        self.dataset_manager = dataset_manager
        self.extractor = VideoExtractor()
        self.worker_thread = None

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)

        # Header Info Card
        header_card = QFrame()
        header_card.setObjectName("card")
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("🎬 Ekstraksi Video ke Kumpulan Frame Gambar")
        title.setObjectName("titleLabel")
        subtitle = QLabel("Potong rekaman video kamera AMR/Robotik menjadi gambar untuk bahan anotasi dataset VLM.")
        subtitle.setObjectName("subtitleLabel")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        main_layout.addWidget(header_card)

        # Settings Group Box
        settings_box = QGroupBox("Pengaturan Input & Output")
        settings_layout = QVBoxLayout(settings_box)
        settings_layout.setSpacing(12)

        # Video Input File
        video_row = QHBoxLayout()
        video_label = QLabel("File Video:")
        video_label.setFixedWidth(130)
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("Pilih file video (MP4, AVI, MOV, MKV)...")
        self.browse_video_btn = QPushButton("Pilih Video...")
        self.browse_video_btn.clicked.connect(self._browse_video)
        video_row.addWidget(video_label)
        video_row.addWidget(self.video_path_edit)
        video_row.addWidget(self.browse_video_btn)
        settings_layout.addLayout(video_row)

        # Output Folder
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

        # File Prefix
        prefix_row = QHBoxLayout()
        prefix_label = QLabel("Prefix Nama File:")
        prefix_label.setFixedWidth(130)
        self.prefix_edit = QLineEdit("amr_frame")
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

    def _browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih File Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;Semua File (*.*)",
        )
        if file_path:
            self.video_path_edit.setText(file_path)
            self.status_label.setText(f"File dipilih: {os.path.basename(file_path)}")

    def _browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Pilih Folder Output Frame")
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _start_extraction(self):
        video_path = self.video_path_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()

        if not video_path or not os.path.isfile(video_path):
            QMessageBox.warning(self, "Peringatan", "Silakan pilih file video yang valid terlebih dahulu.")
            return

        if not output_dir:
            QMessageBox.warning(self, "Peringatan", "Silakan tentukan folder output penyimpanan frame.")
            return

        mode = "frame" if self.radio_frame.isChecked() else "time"
        frame_interval = self.spin_frame.value()
        time_ms = self.spin_time.value()
        max_size = self.spin_resize.value() if self.check_resize.isChecked() else None
        prefix = self.prefix_edit.text().strip() or "amr_frame"

        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Memulai proses ekstraksi frame...")

        self.worker_thread = VideoExtractionThread(
            extractor=self.extractor,
            video_path=video_path,
            output_dir=output_dir,
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

    def _on_progress(self, current, total, saved):
        if total > 0:
            percent = int((current / float(total)) * 100)
            self.progress_bar.setValue(percent)
            self.status_label.setText(
                f"Memproses video: frame {current}/{total} ({percent}%) | Tersimpan: {saved} frame gambar"
            )

    def _on_finished(self, success, message, saved_count):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        if success:
            self.progress_bar.setValue(100)
            self.status_label.setText(f"Selesai! {message}")
            QMessageBox.information(self, "Sukses", f"{message}\nSilakan lanjut ke Menu 2 untuk anotasi.")
            self.frames_extracted.emit(self.output_dir_edit.text().strip())
        else:
            self.status_label.setText(f"Ekstraksi berhenti: {message}")
            QMessageBox.warning(self, "Info", message)
