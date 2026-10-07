import os
import json
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QFileDialog, QDialog, QLineEdit,
    QFormLayout, QMessageBox, QFrame, QApplication
)
from core.dataset_manager import DatasetManager
from core.gemini_client import GeminiClient
from ui.tab_video import TabVideo
from ui.tab_annotate import TabAnnotate
from ui.tab_split import TabSplit
from ui.tab_prompts import TabPrompts
from ui.tab_export import TabExport
from ui.components.styles import DARK_THEME, LIGHT_THEME

SETTINGS_FILE = os.path.expanduser("~/.amr_vlm_settings.json")


class SettingsDialog(QDialog):
    """Dialog to configure Gemini API Key and preferences."""

    def __init__(self, current_key="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pengaturan Gemini AI & API Key")
        self.setFixedWidth(460)
        self.api_key_result = current_key

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        info = QLabel(
            "Masukkan <b>Gemini API Key</b> Anda dari "
            "<a href='https://aistudio.google.com/'>Google AI Studio</a>.<br>"
            "API Key digunakan untuk fitur auto-grounding deteksi objek AMR."
        )
        info.setOpenExternalLinks(True)
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()
        self.key_edit = QLineEdit(current_key)
        self.key_edit.setPlaceholderText("AIzaSy...")
        self.key_edit.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        form.addRow("Gemini API Key:", self.key_edit)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Batal")
        cancel_btn.clicked.connect(self.reject)

        save_btn = QPushButton("Simpan")
        save_btn.setObjectName("primaryBtn")
        save_btn.clicked.connect(self._save)

        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    def _save(self):
        self.api_key_result = self.key_edit.text().strip()
        self.accept()


class MainWindow(QMainWindow):
    """Main Application Window for AMR VLM Dataset Maker."""

    def __init__(self, initial_project_dir: str = None):
        super().__init__()
        self.setWindowTitle("AMR VLM Dataset Maker - AI Grounding & Annotation")
        self.resize(1280, 850)

        # State & Settings
        self.current_theme = "dark"
        self.api_key = ""
        self._load_app_settings()

        # Core Engines
        default_dir = initial_project_dir or os.path.abspath("./amr_dataset_project")
        self.dataset_manager = DatasetManager(default_dir)
        self.gemini_client = GeminiClient(api_key=self.api_key)

        self._init_ui()
        self._apply_theme(self.current_theme)

    def _load_app_settings(self):
        if os.path.isfile(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.api_key = data.get("api_key", "")
                    self.current_theme = data.get("theme", "dark")
            except Exception:
                pass

    def _save_app_settings(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "api_key": self.api_key,
                    "theme": self.current_theme,
                }, f, indent=2)
        except Exception:
            pass

    def _init_ui(self):
        central_widget = QWidget(self)
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(14, 12, 14, 12)
        root_layout.setSpacing(10)

        # ---------------- Top Navigation & Header Bar ----------------
        header = QFrame()
        header.setObjectName("panelCard")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 10, 14, 10)
        header_layout.setSpacing(12)

        # Logo / Title
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        app_title = QLabel("🤖 AMR VLM Dataset Maker")
        app_title.setObjectName("titleLabel")
        app_desc = QLabel("Human-in-the-Loop Vision-Language Dataset Tool for Autonomous Mobile Robots")
        app_desc.setObjectName("subtitleLabel")
        title_box.addWidget(app_title)
        title_box.addWidget(app_desc)
        header_layout.addLayout(title_box)

        header_layout.addStretch()

        # Project Folder Indicator & Switcher
        proj_box = QHBoxLayout()
        self.proj_label = QLabel(f"📁 Project: {os.path.basename(self.dataset_manager.project_dir)}")
        self.proj_label.setStyleSheet("font-weight: 600; color: #38bdf8;")
        self.proj_label.setToolTip(self.dataset_manager.project_dir)

        self.open_proj_btn = QPushButton("Ganti Project...")
        self.open_proj_btn.clicked.connect(self._choose_project_dir)

        proj_box.addWidget(self.proj_label)
        proj_box.addWidget(self.open_proj_btn)
        header_layout.addLayout(proj_box)

        # Settings & Theme Toggle
        self.settings_btn = QPushButton("⚙️ Gemini API")
        self.settings_btn.clicked.connect(self._open_settings)
        header_layout.addWidget(self.settings_btn)

        self.theme_btn = QPushButton("☀️ Light" if self.current_theme == "dark" else "🌙 Dark")
        self.theme_btn.clicked.connect(self._toggle_theme)
        header_layout.addWidget(self.theme_btn)

        root_layout.addWidget(header)

        # ---------------- Main Tab Widget (5 Menus) ----------------
        self.tabs = QTabWidget()

        # Tab 1: Video to Frames
        self.tab_video = TabVideo(self.dataset_manager, self)
        self.tab_video.frames_extracted.connect(self._on_frames_extracted)
        self.tabs.addTab(self.tab_video, "🎬 1. Video to Frames")

        # Tab 2: Frame Annotation & AI Grounding
        self.tab_annotate = TabAnnotate(self.dataset_manager, self.gemini_client, self)
        self.tab_annotate.dataset_updated.connect(self._on_dataset_updated)
        self.tabs.addTab(self.tab_annotate, "🏷️ 2. Anotasi Objek (AI Grounding)")

        # Tab 3: Dataset Split Partitioning (Train / Val / Test)
        self.tab_split = TabSplit(self.dataset_manager, self)
        self.tab_split.dataset_updated.connect(self._on_dataset_updated)
        self.tabs.addTab(self.tab_split, "✂️ 3. Split Dataset")

        # Tab 4: Prompt Variations & Template Manager (Train / Val / Test)
        self.tab_prompts = TabPrompts(self.dataset_manager, self)
        self.tab_prompts.dataset_updated.connect(self._on_dataset_updated)
        self.tabs.addTab(self.tab_prompts, "💬 4. Prompt Editor")

        # Tab 5: Final Dataset Exporter (train.jsonl, val.jsonl, test.jsonl)
        self.tab_export = TabExport(self.dataset_manager, self)
        self.tab_export.dataset_updated.connect(self._on_dataset_updated)
        self.tabs.addTab(self.tab_export, "📦 5. Ekspor Dataset")

        self.tabs.currentChanged.connect(self._on_tab_changed)

        root_layout.addWidget(self.tabs)

        # Initial data load
        self.tab_annotate.reload_images()

    def _choose_project_dir(self):
        new_dir = QFileDialog.getExistingDirectory(
            self,
            "Pilih atau Buat Folder Project Dataset",
            self.dataset_manager.project_dir,
        )
        if new_dir:
            self.dataset_manager.set_project_dir(new_dir)
            self.proj_label.setText(f"📁 Project: {os.path.basename(new_dir)}")
            self.proj_label.setToolTip(new_dir)
            self.tab_video.update_output_dir(self.dataset_manager.frames_dir)
            self.tab_annotate.reload_images()
            self.tab_split.reload_data()
            self.tab_prompts.reload_data()
            self.tab_export.reload_data()
            QMessageBox.information(self, "Project Dimuat", f"Project aktif diubah ke:\n{new_dir}")

    def _open_settings(self):
        dialog = SettingsDialog(current_key=self.gemini_client.api_key, parent=self)
        if dialog.exec():
            self.api_key = dialog.api_key_result
            self.gemini_client.set_api_key(self.api_key)
            self._save_app_settings()
            QMessageBox.information(self, "Pengaturan Disimpan", "Gemini API Key berhasil disimpan.")

    def _toggle_theme(self):
        if self.current_theme == "dark":
            self.current_theme = "light"
            self.theme_btn.setText("🌙 Dark")
        else:
            self.current_theme = "dark"
            self.theme_btn.setText("☀️ Light")

        self._apply_theme(self.current_theme)
        self._save_app_settings()

    def _apply_theme(self, theme_name: str):
        app = QApplication.instance()
        if theme_name == "dark":
            app.setStyleSheet(DARK_THEME)
        else:
            app.setStyleSheet(LIGHT_THEME)

    def _on_tab_changed(self, index: int):
        if index == 1:
            self.tab_annotate.reload_images()
        elif index == 2:
            self.tab_split.reload_data()
        elif index == 3:
            self.tab_prompts.reload_data()
        elif index == 4:
            self.tab_export.reload_data()

    def _on_frames_extracted(self, output_dir: str):
        # Auto-switch to Tab 2 and point to the extracted folder
        self.dataset_manager.set_active_frames_dir(output_dir)
        self.tab_annotate.reload_images()
        self.tabs.setCurrentIndex(1)

    def _on_dataset_updated(self):
        # Refresh current visible tab if needed
        curr = self.tabs.currentIndex()
        if curr == 2:
            self.tab_split.reload_data()
        elif curr == 3:
            self.tab_prompts.reload_data()
        elif curr == 4:
            self.tab_export.reload_data()
