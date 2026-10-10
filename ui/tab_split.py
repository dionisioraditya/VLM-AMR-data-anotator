import os
from typing import List, Dict, Any, Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QSplitter, QFrame, QGroupBox,
    QMessageBox, QSpinBox, QCheckBox, QSlider, QLineEdit
)


class TabSplit(QWidget):
    """Menu 3: Dataset Partitioning Editor (Train, Validation, Test Split)."""
    dataset_updated = Signal()

    def __init__(self, dataset_manager, parent=None):
        super().__init__(parent)
        self.dataset_manager = dataset_manager
        self.selected_image: Optional[str] = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(12)

        # 1. Header Card
        header_card = QFrame()
        header_card.setObjectName("panelCard")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(16, 12, 16, 12)

        title_col = QVBoxLayout()
        title = QLabel("✂️ Split Dataset (Train / Validation / Test)")
        title.setObjectName("titleLabel")
        subtitle = QLabel("Partisi frame hasil anotasi menjadi subset pelatihan, validasi, dan pengujian bebas data leakage.")
        subtitle.setObjectName("subtitleLabel")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header_layout.addLayout(title_col)

        header_layout.addStretch()

        self.stats_badge = QLabel("0 Frame Total")
        self.stats_badge.setObjectName("badgeLabel")
        header_layout.addWidget(self.stats_badge)

        main_layout.addWidget(header_card)

        # 2. Auto-Split Configuration Card
        ctrl_card = QFrame()
        ctrl_card.setObjectName("card")
        ctrl_layout = QVBoxLayout(ctrl_card)
        ctrl_layout.setContentsMargins(14, 12, 14, 12)
        ctrl_layout.setSpacing(10)

        ctrl_header_row = QHBoxLayout()
        ctrl_title = QLabel("⚙️ Konfigurasi Proporsi Auto-Split:")
        ctrl_title.setStyleSheet("font-weight: 700; color: #38bdf8;")
        ctrl_header_row.addWidget(ctrl_title)

        self.role_mode_label = QLabel("")
        self.role_mode_label.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")
        ctrl_header_row.addStretch()
        ctrl_header_row.addWidget(self.role_mode_label)
        ctrl_layout.addLayout(ctrl_header_row)

        ratios_row = QHBoxLayout()
        ratios_row.setSpacing(16)

        # Train Ratio
        tr_box = QHBoxLayout()
        tr_lbl = QLabel("🟢 Train:")
        tr_lbl.setStyleSheet("font-weight: 600; color: #22c55e;")
        self.train_spin = QSpinBox()
        self.train_spin.setRange(0, 100)
        self.train_spin.setValue(70)
        self.train_spin.setSuffix("%")
        self.train_spin.valueChanged.connect(self._on_ratio_changed)
        tr_box.addWidget(tr_lbl)
        tr_box.addWidget(self.train_spin)
        ratios_row.addLayout(tr_box)

        # Val Ratio
        val_box = QHBoxLayout()
        val_lbl = QLabel("🟡 Validation:")
        val_lbl.setStyleSheet("font-weight: 600; color: #f59e0b;")
        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 100)
        self.val_spin.setValue(15)
        self.val_spin.setSuffix("%")
        self.val_spin.valueChanged.connect(self._on_ratio_changed)
        val_box.addWidget(val_lbl)
        val_box.addWidget(self.val_spin)
        ratios_row.addLayout(val_box)

        # Test Ratio
        test_box = QHBoxLayout()
        self.test_lbl = QLabel("🔵 Test:")
        self.test_lbl.setStyleSheet("font-weight: 600; color: #38bdf8;")
        self.test_spin = QSpinBox()
        self.test_spin.setRange(0, 100)
        self.test_spin.setValue(15)
        self.test_spin.setSuffix("%")
        self.test_spin.valueChanged.connect(self._on_ratio_changed)
        test_box.addWidget(self.test_lbl)
        test_box.addWidget(self.test_spin)
        ratios_row.addLayout(test_box)

        self.total_ratio_label = QLabel("Total: 100%")
        self.total_ratio_label.setStyleSheet("font-weight: 700; color: #22c55e;")
        ratios_row.addWidget(self.total_ratio_label)

        ratios_row.addStretch()

        self.stratify_check = QCheckBox("Seimbangkan Frame Positif & Negatif (Stratified)")
        self.stratify_check.setChecked(True)
        self.stratify_check.setToolTip("Menjaga rasio frame berobjek dan frame negatif (0 box) seimbang di setiap split")
        ratios_row.addWidget(self.stratify_check)

        self.run_auto_split_btn = QPushButton("🎲 Jalankan Auto-Split")
        self.run_auto_split_btn.setObjectName("primaryBtn")
        self.run_auto_split_btn.clicked.connect(self._run_auto_split)
        ratios_row.addWidget(self.run_auto_split_btn)

        ctrl_layout.addLayout(ratios_row)
        main_layout.addWidget(ctrl_card)

        # 3. Main Splitter: 3 Partition Columns + Preview
        splitter = QSplitter(Qt.Horizontal)

        # Column 1: Train
        self.train_col, self.train_list, self.train_badge = self._create_split_column(
            "train", "🟢 Train Set", "#22c55e", [
                ("➡️ Pindah ke Val", lambda: self._move_selected("train", "val")),
                ("➡️ Pindah ke Test", lambda: self._move_selected("train", "test")),
            ]
        )
        splitter.addWidget(self.train_col)

        # Column 2: Validation
        self.val_col, self.val_list, self.val_badge = self._create_split_column(
            "val", "🟡 Validation Set", "#f59e0b", [
                ("⬅️ Pindah ke Train", lambda: self._move_selected("val", "train")),
                ("➡️ Pindah ke Test", lambda: self._move_selected("val", "test")),
            ]
        )
        splitter.addWidget(self.val_col)

        # Column 3: Test
        self.test_col, self.test_list, self.test_badge = self._create_split_column(
            "test", "🔵 Test Set", "#38bdf8", [
                ("⬅️ Pindah ke Train", lambda: self._move_selected("test", "train")),
                ("⬅️ Pindah ke Val", lambda: self._move_selected("test", "val")),
            ]
        )
        splitter.addWidget(self.test_col)

        # Column 4: Quick Preview
        preview_panel = QFrame()
        preview_panel.setObjectName("panelCard")
        preview_panel.setMinimumWidth(260)
        preview_panel.setMaximumWidth(320)
        prev_layout = QVBoxLayout(preview_panel)
        prev_layout.setContentsMargins(12, 12, 12, 12)
        prev_layout.setSpacing(8)

        prev_title = QLabel("🖼️ Detail Frame Terpilih")
        prev_title.setStyleSheet("font-weight: 700; color: #cbd5e1;")
        prev_layout.addWidget(prev_title)

        self.preview_img_label = QLabel("Pilih frame di daftar untuk preview")
        self.preview_img_label.setAlignment(Qt.AlignCenter)
        self.preview_img_label.setStyleSheet("background-color: #0f172a; border-radius: 6px; min-height: 180px; color: #64748b;")
        prev_layout.addWidget(self.preview_img_label)

        self.preview_info_label = QLabel("")
        self.preview_info_label.setWordWrap(True)
        self.preview_info_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        prev_layout.addWidget(self.preview_info_label)

        prev_layout.addStretch()

        self.save_splits_btn = QPushButton("💾 Simpan Pembagian Split")
        self.save_splits_btn.setObjectName("successBtn")
        self.save_splits_btn.setFixedHeight(36)
        self.save_splits_btn.clicked.connect(self._save_splits)
        prev_layout.addWidget(self.save_splits_btn)

        splitter.addWidget(preview_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        splitter.setStretchFactor(3, 0)

        main_layout.addWidget(splitter)

    def _create_split_column(self, split_key: str, title: str, color_hex: str, actions: list):
        col_frame = QFrame()
        col_frame.setObjectName("panelCard")
        col_layout = QVBoxLayout(col_frame)
        col_layout.setContentsMargins(10, 10, 10, 10)
        col_layout.setSpacing(8)

        # Header with badge
        hdr_row = QHBoxLayout()
        hdr_lbl = QLabel(title)
        hdr_lbl.setStyleSheet(f"font-weight: 700; color: {color_hex}; font-size: 13px;")
        badge = QLabel("0 Frame")
        badge.setStyleSheet(f"background-color: {color_hex}22; color: {color_hex}; border: 1px solid {color_hex}; padding: 2px 6px; border-radius: 4px; font-size: 11px;")
        hdr_row.addWidget(hdr_lbl)
        hdr_row.addStretch()
        hdr_row.addWidget(badge)
        col_layout.addLayout(hdr_row)

        # Search box
        search_edit = QLineEdit()
        search_edit.setPlaceholderText("Cari frame...")
        col_layout.addWidget(search_edit)

        # List
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        col_layout.addWidget(list_widget)

        # Search filter connection
        search_edit.textChanged.connect(lambda txt: self._filter_list(list_widget, txt))

        # Preview on click
        list_widget.currentRowChanged.connect(lambda row: self._on_frame_clicked(list_widget, row))

        # Action Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        for btn_text, btn_cb in actions:
            btn = QPushButton(btn_text)
            btn.setFixedHeight(28)
            btn.setStyleSheet("font-size: 11px;")
            btn.clicked.connect(btn_cb)
            btn_row.addWidget(btn)
        col_layout.addLayout(btn_row)

        return col_frame, list_widget, badge

    def _filter_list(self, list_widget: QListWidget, query: str):
        q = query.strip().lower()
        for idx in range(list_widget.count()):
            item = list_widget.item(idx)
            item.setHidden(q not in item.text().lower() if q else False)

    def reload_data(self):
        """Reload all splits and statistics from dataset_manager."""
        all_imgs = self.dataset_manager.get_image_list()
        splits = self.dataset_manager.get_splits()
        role_counts = (
            self.dataset_manager.get_role_counts()
            if hasattr(self.dataset_manager, "get_role_counts")
            else {"train_val": len(all_imgs), "test": 0}
        )
        test_role_cnt = role_counts.get("test", 0)
        tv_role_cnt = role_counts.get("train_val", 0)

        if test_role_cnt > 0:
            self.role_mode_label.setText(
                f"📌 Video Test Terpisah: {test_role_cnt} frame [Test Only] ➡️ 100% Test | "
                f"{tv_role_cnt} frame [Train & Val] dibagi Train/Val"
            )
            if self.test_spin.isEnabled():
                self.train_spin.blockSignals(True)
                self.val_spin.blockSignals(True)
                self.test_spin.blockSignals(True)
                cur_val = self.val_spin.value()
                self.test_spin.setValue(0)
                self.test_spin.setEnabled(False)
                self.test_spin.setToolTip("Dikunci otomatis karena Test Set diambil 100% dari video berkategori 'Test Only'.")
                self.train_spin.setValue(max(0, 100 - cur_val))
                self.train_spin.blockSignals(False)
                self.val_spin.blockSignals(False)
                self.test_spin.blockSignals(False)
                self._on_ratio_changed()
        else:
            self.role_mode_label.setText("")
            if not self.test_spin.isEnabled():
                self.train_spin.blockSignals(True)
                self.val_spin.blockSignals(True)
                self.test_spin.blockSignals(True)
                self.test_spin.setEnabled(True)
                self.test_spin.setToolTip("")
                self.train_spin.setValue(70)
                self.val_spin.setValue(15)
                self.test_spin.setValue(15)
                self.train_spin.blockSignals(False)
                self.val_spin.blockSignals(False)
                self.test_spin.blockSignals(False)
                self._on_ratio_changed()

        self.train_list.blockSignals(True)
        self.val_list.blockSignals(True)
        self.test_list.blockSignals(True)

        self.train_list.clear()
        self.val_list.clear()
        self.test_list.clear()

        annotated_count = 0
        negative_count = 0

        def populate_col(img_list: List[str], target_widget: QListWidget):
            nonlocal annotated_count, negative_count
            pos = 0
            neg = 0
            for img in img_list:
                anno = self.dataset_manager.get_annotation(img)
                b_count = len(anno.get("boxes", []))
                role = (
                    self.dataset_manager.get_frame_role(img)
                    if hasattr(self.dataset_manager, "get_frame_role")
                    else "train_val"
                )
                role_tag = "[Test]" if role == "test" else "[T&V]"
                if b_count > 0:
                    pos += 1
                    annotated_count += 1
                    icon = "🟢"
                    text = f"{icon} {role_tag} [{b_count} box] {img}"
                else:
                    neg += 1
                    negative_count += 1
                    icon = "⚪"
                    text = f"{icon} {role_tag} [0 box] {img}"
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, img)
                target_widget.addItem(item)
            return pos, neg

        tr_pos, tr_neg = populate_col(splits.get("train", []), self.train_list)
        va_pos, va_neg = populate_col(splits.get("val", []), self.val_list)
        te_pos, te_neg = populate_col(splits.get("test", []), self.test_list)

        self.train_list.blockSignals(False)
        self.val_list.blockSignals(False)
        self.test_list.blockSignals(False)

        tr_total = tr_pos + tr_neg
        va_total = va_pos + va_neg
        te_total = te_pos + te_neg
        all_total = len(all_imgs)

        self.train_badge.setText(f"{tr_total} Frame ({tr_pos} Pos • {tr_neg} Neg)")
        self.val_badge.setText(f"{va_total} Frame ({va_pos} Pos • {va_neg} Neg)")
        self.test_badge.setText(f"{te_total} Frame ({te_pos} Pos • {te_neg} Neg)")

        folder_name = os.path.basename(self.dataset_manager.active_frames_dir or "frames")
        self.stats_badge.setText(f"📁 {folder_name} | {all_total} Frame ({annotated_count} Ada Objek • {negative_count} Negatif)")

    def _on_ratio_changed(self):
        total = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        self.total_ratio_label.setText(f"Total: {total}%")
        if total == 100:
            self.total_ratio_label.setStyleSheet("font-weight: 700; color: #22c55e;")
            self.run_auto_split_btn.setEnabled(True)
        else:
            self.total_ratio_label.setStyleSheet("font-weight: 700; color: #ef4444;")
            self.run_auto_split_btn.setEnabled(False)

    def _run_auto_split(self):
        total = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        if total != 100:
            QMessageBox.warning(self, "Peringatan Rasio", "Total persentase harus tepat 100%.")
            return

        r_tr = self.train_spin.value() / 100.0
        r_va = self.val_spin.value() / 100.0
        r_te = self.test_spin.value() / 100.0
        stratify = self.stratify_check.isChecked()

        splits = self.dataset_manager.auto_split(
            train_ratio=r_tr,
            val_ratio=r_va,
            test_ratio=r_te,
            stratify=stratify,
        )
        self.reload_data()
        self.dataset_updated.emit()

        has_test_role = (
            self.dataset_manager.has_test_only_frames()
            if hasattr(self.dataset_manager, "has_test_only_frames")
            else False
        )
        if has_test_role:
            msg = (
                f"Pembagian dataset berhasil diperbarui berdasarkan kategori video!\n\n"
                f"• 🟢 Train Set: {len(splits.get('train', []))} frame (dari video Train & Val)\n"
                f"• 🟡 Validation Set: {len(splits.get('val', []))} frame (dari video Train & Val)\n"
                f"• 🔵 Test Set: {len(splits.get('test', []))} frame (100% dari video Test Only)"
            )
        else:
            msg = "Pembagian dataset berhasil diperbarui!"
        QMessageBox.information(self, "Auto-Split Sukses 🎉", msg)

    def _move_selected(self, source_split: str, target_split: str):
        src_widget = {"train": self.train_list, "val": self.val_list, "test": self.test_list}.get(source_split)
        if not src_widget:
            return

        selected_items = src_widget.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "Pilih Frame", f"Pilih minimal satu frame di kolom {source_split.upper()} untuk dipindahkan.")
            return

        splits = self.dataset_manager.get_splits()
        for it in selected_items:
            img = it.data(Qt.UserRole)
            if img in splits[source_split]:
                splits[source_split].remove(img)
            if img not in splits[target_split]:
                splits[target_split].append(img)

        self.dataset_manager.set_splits(splits)
        self.reload_data()
        self.dataset_updated.emit()

    def _on_frame_clicked(self, list_widget: QListWidget, row: int):
        if row < 0:
            return
        item = list_widget.item(row)
        if not item:
            return
        img_name = item.data(Qt.UserRole)
        self.selected_image = img_name

        img_path = self.dataset_manager.get_image_path(img_name)
        if os.path.isfile(img_path):
            pix = QPixmap(img_path)
            scaled = pix.scaled(280, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_img_label.setPixmap(scaled)
        else:
            self.preview_img_label.setText("Gambar tidak ditemukan")

        anno = self.dataset_manager.get_annotation(img_name)
        boxes = anno.get("boxes", [])
        prompts = anno.get("prompts", [])
        split_name = self.dataset_manager.get_split_for_image(img_name).upper()
        role = (
            self.dataset_manager.get_frame_role(img_name)
            if hasattr(self.dataset_manager, "get_frame_role")
            else "train_val"
        )
        role_display = "🔵 Test Only" if role == "test" else "🟢 Train & Val"

        info_lines = [
            f"<b>File:</b> {img_name}",
            f"<b>Kategori Video:</b> {role_display}",
            f"<b>Partisi Saat Ini:</b> {split_name}",
            f"<b>Jumlah Box:</b> {len(boxes)} {'(Negatif)' if not boxes else ''}",
            f"<b>Jumlah Prompt:</b> {len(prompts)}",
        ]
        if boxes:
            box_desc = ", ".join(f"{b.get('label', 'obj')}" for b in boxes[:4])
            if len(boxes) > 4:
                box_desc += "..."
            info_lines.append(f"<b>Objek:</b> {box_desc}")

        self.preview_info_label.setText("<br>".join(info_lines))

    def _save_splits(self):
        # Trigger explicit save to project_config.json
        splits = self.dataset_manager.get_splits()
        self.dataset_manager.set_splits(splits)
        self.dataset_updated.emit()
        QMessageBox.information(self, "Tersimpan", "Pembagian partisi dataset berhasil disimpan ke konfigurasi proyek!")
