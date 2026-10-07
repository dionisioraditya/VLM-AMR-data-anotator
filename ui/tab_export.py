import os
import subprocess
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QCheckBox, QFileDialog, QMessageBox,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox
)
from core.exporter import DatasetExporter


class TabExport(QWidget):
    """Menu 5: Final Dataset Export Dashboard (Train, Validation, Test to JSONL)."""
    dataset_updated = Signal()

    def __init__(self, dataset_manager, parent=None):
        super().__init__(parent)
        self.dataset_manager = dataset_manager
        self.last_export_dir: str = ""
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)

        # 1. Header Card
        header_card = QFrame()
        header_card.setObjectName("panelCard")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(16, 12, 16, 12)

        title_col = QVBoxLayout()
        title = QLabel("📦 Ekspor Dataset VLM Siap Latih")
        title.setObjectName("titleLabel")
        subtitle = QLabel("Kompilasi dataset terpartisi (Train, Validation, Test) ke format standar Qwen2-VL, PaliGemma 2, atau ShareGPT.")
        subtitle.setObjectName("subtitleLabel")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header_layout.addLayout(title_col)

        header_layout.addStretch()

        self.summary_badge = QLabel("0 Total Sampel")
        self.summary_badge.setObjectName("badgeLabel")
        header_layout.addWidget(self.summary_badge)

        main_layout.addWidget(header_card)

        # 2. Rekapitulasi Data Table Card
        recap_card = QFrame()
        recap_card.setObjectName("card")
        recap_layout = QVBoxLayout(recap_card)
        recap_layout.setContentsMargins(16, 14, 16, 14)
        recap_layout.setSpacing(10)

        recap_title = QLabel("📊 Rekapitulasi Pembagian Dataset Siap Ekspor:")
        recap_title.setStyleSheet("font-weight: 700; color: #38bdf8;")
        recap_layout.addWidget(recap_title)

        self.recap_table = QTableWidget(4, 5)
        self.recap_table.setHorizontalHeaderLabels([
            "Partisi Dataset", "Jumlah Frame", "Sampel Positif (Target)", "Sampel Negatif (Frontier)", "Total Sampel JSONL"
        ])
        self.recap_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.recap_table.verticalHeader().setVisible(False)
        self.recap_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.recap_table.setFixedHeight(150)
        recap_layout.addWidget(self.recap_table)

        main_layout.addWidget(recap_card)

        # 3. Export Configurations Box
        export_box = QGroupBox("⚙️ Pengaturan & Format Ekspor")
        export_layout = QVBoxLayout(export_box)
        export_layout.setContentsMargins(16, 14, 16, 14)
        export_layout.setSpacing(12)

        # Row 1: Format & Options
        row1 = QHBoxLayout()
        fmt_lbl = QLabel("Format Model:")
        fmt_lbl.setStyleSheet("font-weight: 600; color: #cbd5e1;")
        self.format_combo = QComboBox()
        self.format_combo.addItem("Qwen2-VL (Grounding + Frontier JSON)", "qwen")
        self.format_combo.addItem("PaliGemma 2 (Hugging Face Prefix-Suffix / <loc>)", "paligemma")
        self.format_combo.addItem("ShareGPT (Conversational Multimodal)", "sharegpt")
        self.format_combo.setFixedWidth(380)

        self.copy_images_check = QCheckBox("Salin file gambar ke subfolder 'images/' (Portabel)")
        self.copy_images_check.setChecked(True)
        self.copy_images_check.setToolTip("Jika dicentang, file gambar disalin sehingga dataset bisa dipindahkan ke GPU server/Colab secara utuh.")

        row1.addWidget(fmt_lbl)
        row1.addWidget(self.format_combo)
        row1.addSpacing(20)
        row1.addWidget(self.copy_images_check)
        row1.addStretch()
        export_layout.addLayout(row1)

        # Row 2: Destination Folder
        row2 = QHBoxLayout()
        out_lbl = QLabel("Folder Tujuan:")
        out_lbl.setStyleSheet("font-weight: 600; color: #cbd5e1;")
        out_lbl.setFixedWidth(100)
        self.export_dir_edit = QLineEdit()
        self.export_dir_edit.setPlaceholderText("Pilih folder tujuan ekspor...")
        self.browse_export_btn = QPushButton("Pilih Folder...")
        self.browse_export_btn.clicked.connect(self._browse_export_dir)

        row2.addWidget(out_lbl)
        row2.addWidget(self.export_dir_edit)
        row2.addWidget(self.browse_export_btn)
        export_layout.addLayout(row2)

        # Row 3: Action Buttons
        row3 = QHBoxLayout()
        self.open_folder_btn = QPushButton("📂 Buka Folder Hasil Ekspor")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._open_export_folder)
        row3.addWidget(self.open_folder_btn)

        row3.addStretch()

        self.export_btn = QPushButton("✨ Ekspor Seluruh Dataset (train, val, test.jsonl)")
        self.export_btn.setObjectName("successBtn")
        self.export_btn.setFixedHeight(40)
        self.export_btn.setStyleSheet("font-weight: bold; font-size: 13px; padding: 0 20px;")
        self.export_btn.clicked.connect(self._run_export)
        row3.addWidget(self.export_btn)

        export_layout.addLayout(row3)
        main_layout.addWidget(export_box)

        main_layout.addStretch()

    def reload_data(self):
        """Update recap table with statistics from dataset_manager."""
        if not self.export_dir_edit.text() and self.dataset_manager.project_dir:
            self.export_dir_edit.setText(os.path.join(self.dataset_manager.project_dir, "exports"))

        splits = self.dataset_manager.get_splits()

        def compute_split_stats(img_list):
            pos_prompts = 0
            neg_prompts = 0
            for img in img_list:
                anno = self.dataset_manager.get_annotation(img)
                prompts = anno.get("prompts", [])
                boxes = anno.get("boxes", [])
                for p in prompts:
                    a_text = p.get("assistant", "").strip().lower()
                    if '"target_detected": true' in a_text or (boxes and "null" not in a_text and '"target_detected": false' not in a_text):
                        pos_prompts += 1
                    else:
                        neg_prompts += 1
            return len(img_list), pos_prompts, neg_prompts, (pos_prompts + neg_prompts)

        tr_imgs, tr_pos, tr_neg, tr_tot = compute_split_stats(splits.get("train", []))
        va_imgs, va_pos, va_neg, va_tot = compute_split_stats(splits.get("val", []))
        te_imgs, te_pos, te_neg, te_tot = compute_split_stats(splits.get("test", []))

        all_imgs = tr_imgs + va_imgs + te_imgs
        all_pos = tr_pos + va_pos + te_pos
        all_neg = tr_neg + va_neg + te_neg
        all_tot = tr_tot + va_tot + te_tot

        rows_data = [
            ("🟢 Train Set (train.jsonl)", tr_imgs, tr_pos, tr_neg, tr_tot),
            ("🟡 Validation Set (val.jsonl)", va_imgs, va_pos, va_neg, va_tot),
            ("🔵 Test Set (test.jsonl)", te_imgs, te_pos, te_neg, te_tot),
            ("📊 TOTAL SEMUA PARTISI", all_imgs, all_pos, all_neg, all_tot),
        ]

        for r_idx, (label, n_img, n_pos, n_neg, n_tot) in enumerate(rows_data):
            self.recap_table.setItem(r_idx, 0, QTableWidgetItem(label))
            self.recap_table.setItem(r_idx, 1, QTableWidgetItem(f"{n_img} Frame"))
            self.recap_table.setItem(r_idx, 2, QTableWidgetItem(f"{n_pos} Prompt"))
            self.recap_table.setItem(r_idx, 3, QTableWidgetItem(f"{n_neg} Prompt"))
            self.recap_table.setItem(r_idx, 4, QTableWidgetItem(f"{n_tot} Sampel"))

            # Highlight total row
            if r_idx == 3:
                for c in range(5):
                    it = self.recap_table.item(r_idx, c)
                    it.setBackground(Qt.darkGray)

        self.summary_badge.setText(f"{all_tot} Total Sampel Siap Ekspor")

    def _browse_export_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Pilih Folder Tujuan Ekspor")
        if d:
            self.export_dir_edit.setText(d)

    def _open_export_folder(self):
        if self.last_export_dir and os.path.isdir(self.last_export_dir):
            try:
                subprocess.run(["xdg-open", self.last_export_dir], check=False)
            except Exception:
                pass

    def _run_export(self):
        output_dir = self.export_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Peringatan", "Silakan tentukan folder tujuan ekspor terlebih dahulu.")
            return

        fmt = self.format_combo.currentData()
        copy_img = self.copy_images_check.isChecked()

        success, msg, stats = DatasetExporter.export(
            dataset_manager=self.dataset_manager,
            output_dir=output_dir,
            format_type=fmt,
            copy_images=copy_img,
        )

        if success:
            self.last_export_dir = output_dir
            self.open_folder_btn.setEnabled(True)
            QMessageBox.information(self, "Ekspor Berhasil 🎉", f"{msg}\n\nLokasi:\n{output_dir}")
        else:
            QMessageBox.warning(self, "Gagal Ekspor", msg)
