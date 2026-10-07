import os
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QMessageBox,
    QTabWidget, QWidget, QInputDialog, QFrame
)
from core.prompt_generator import TemplatePromptGenerator


class TemplateManagerDialog(QDialog):
    """Dialog to manage, add, edit, delete, and reset prompt templates for a specific dataset split."""
    templates_updated = Signal(str)  # split_name

    def __init__(self, template_file_path: str, split_name: str = "train", parent=None):
        super().__init__(parent)
        self.template_file_path = template_file_path
        self.split_name = split_name.lower().strip()
        self.prompt_gen = TemplatePromptGenerator(template_file_path=self.template_file_path, split=self.split_name)

        split_label = self.split_name.capitalize()
        self.setWindowTitle(f"⚙️ Kelola Template Prompt - {split_label}")
        self.setFixedWidth(540)
        self.setFixedHeight(620)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        # Header Title
        split_badge_color = {
            "train": "#22c55e",
            "val": "#f59e0b",
            "validation": "#f59e0b",
            "test": "#38bdf8",
        }.get(self.split_name, "#38bdf8")

        title_row = QHBoxLayout()
        title = QLabel(f"⚙️ Kelola Template Prompt")
        title.setObjectName("titleLabel")
        badge = QLabel(f"Partisi: {self.split_name.upper()}")
        badge.setStyleSheet(f"background-color: {split_badge_color}22; color: {split_badge_color}; border: 1px solid {split_badge_color}; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;")
        title_row.addWidget(title)
        title_row.addWidget(badge)
        title_row.addStretch()
        layout.addLayout(title_row)

        desc = QLabel(
            "Setiap kalimat template <b>wajib memuat huruf <code>X</code></b> sebagai placeholder nama objek.<br>"
            "Contoh: <i>'Tolong cari <b>X</b> di ruangan ini.'</i> atau <i>'Find the <b>X</b> near the robot.'</i>"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #94a3b8; font-size: 12px; line-height: 1.4;")
        layout.addWidget(desc)

        # Tabs: ID vs EN
        self.tab_widget = QTabWidget()

        # Tab ID
        tab_id = QWidget()
        id_layout = QVBoxLayout(tab_id)
        id_layout.setContentsMargins(8, 8, 8, 8)
        self.id_list = QListWidget()
        id_layout.addWidget(self.id_list)
        self.tab_widget.addTab(tab_id, "🇮🇩 Bahasa Indonesia")

        # Tab EN
        tab_en = QWidget()
        en_layout = QVBoxLayout(tab_en)
        en_layout.setContentsMargins(8, 8, 8, 8)
        self.en_list = QListWidget()
        en_layout.addWidget(self.en_list)
        self.tab_widget.addTab(tab_en, "🇬🇧 English")

        layout.addWidget(self.tab_widget)

        # Action Buttons (Add, Edit, Delete)
        btn_action_row = QHBoxLayout()
        self.add_btn = QPushButton("➕ Tambah Template")
        self.add_btn.setObjectName("primaryBtn")
        self.add_btn.clicked.connect(self._add_template)

        self.edit_btn = QPushButton("✏️ Edit")
        self.edit_btn.clicked.connect(self._edit_template)

        self.del_btn = QPushButton("🗑️ Hapus")
        self.del_btn.setObjectName("dangerBtn")
        self.del_btn.clicked.connect(self._delete_template)

        btn_action_row.addWidget(self.add_btn)
        btn_action_row.addWidget(self.edit_btn)
        btn_action_row.addWidget(self.del_btn)
        layout.addLayout(btn_action_row)

        # Footer row: Reset to default + Close
        footer_row = QHBoxLayout()
        self.reset_btn = QPushButton("🔄 Reset ke Default Master")
        self.reset_btn.setToolTip("Kembalikan daftar template split ini ke pengaturan master bawaan aplikasi")
        self.reset_btn.clicked.connect(self._reset_to_default)
        footer_row.addWidget(self.reset_btn)

        footer_row.addStretch()

        self.save_btn = QPushButton("💾 Simpan & Terapkan")
        self.save_btn.setObjectName("successBtn")
        self.save_btn.clicked.connect(self._save_and_close)
        footer_row.addWidget(self.save_btn)

        layout.addLayout(footer_row)

        self._populate_lists()

    def _get_current_list_and_lang(self):
        if self.tab_widget.currentIndex() == 0:
            return self.id_list, "id"
        return self.en_list, "en"

    def _populate_lists(self):
        self.id_list.clear()
        for tpl in self.prompt_gen.id_templates:
            self.id_list.addItem(tpl)

        self.en_list.clear()
        for tpl in self.prompt_gen.en_templates:
            self.en_list.addItem(tpl)

        self.tab_widget.setTabText(0, f"🇮🇩 Bahasa Indonesia ({len(self.prompt_gen.id_templates)})")
        self.tab_widget.setTabText(1, f"🇬🇧 English ({len(self.prompt_gen.en_templates)})")

    def _add_template(self):
        list_widget, lang = self._get_current_list_and_lang()
        lang_name = "Bahasa Indonesia" if lang == "id" else "English"
        text, ok = QInputDialog.getText(
            self,
            f"Tambah Template ({lang_name})",
            "Masukkan kalimat template baru (wajib mengandung 'X'):",
            QLineEdit.Normal,
            "Cari X pada citra ini." if lang == "id" else "Find X in this image."
        )
        if ok and text:
            clean_text = text.strip()
            if "X" not in clean_text:
                QMessageBox.warning(self, "Peringatan", "Kalimat template wajib memuat huruf besar 'X' sebagai placeholder target.")
                return
            success = self.prompt_gen.add_template(clean_text, lang=lang)
            if success:
                self._populate_lists()
            else:
                QMessageBox.information(self, "Info", "Template tersebut sudah ada di daftar.")

    def _edit_template(self):
        list_widget, lang = self._get_current_list_and_lang()
        item = list_widget.currentItem()
        if not item:
            QMessageBox.information(self, "Pilih Template", "Pilih template di daftar terlebih dahulu untuk diedit.")
            return

        old_text = item.text()
        text, ok = QInputDialog.getText(
            self,
            "Edit Template",
            "Ubah kalimat template (wajib mengandung 'X'):",
            QLineEdit.Normal,
            old_text
        )
        if ok and text:
            clean_text = text.strip()
            if "X" not in clean_text:
                QMessageBox.warning(self, "Peringatan", "Kalimat template wajib memuat huruf besar 'X'.")
                return
            self.prompt_gen.edit_template(old_text, clean_text, lang=lang)
            self._populate_lists()

    def _delete_template(self):
        list_widget, lang = self._get_current_list_and_lang()
        row = list_widget.currentRow()
        if row < 0:
            QMessageBox.information(self, "Pilih Template", "Pilih template yang ingin dihapus.")
            return

        item = list_widget.item(row)
        text = item.text()
        reply = QMessageBox.question(
            self,
            "Konfirmasi Hapus",
            f"Hapus template berikut?\n\n\"{text}\"",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.prompt_gen.remove_template(text, lang=lang)
            self._populate_lists()

    def _reset_to_default(self):
        reply = QMessageBox.question(
            self,
            "Konfirmasi Reset",
            f"Apakah Anda yakin ingin mengembalikan seluruh template partisi '{self.split_name}' ke bawaan master aplikasi?\n"
            "Perubahan kustom Anda pada split ini akan ditimpa dengan template default.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        master_path = TemplatePromptGenerator.get_default_template_path_for_split(self.split_name)
        if os.path.isfile(master_path):
            import shutil
            shutil.copy2(master_path, self.template_file_path)
            self.prompt_gen.reload()
            self._populate_lists()
            QMessageBox.information(self, "Reset Sukses", f"Template '{self.split_name}' berhasil dikembalikan ke bawaan.")

    def _save_and_close(self):
        self.prompt_gen.save_templates_to_file(self.template_file_path)
        self.templates_updated.emit(self.split_name)
        self.accept()
