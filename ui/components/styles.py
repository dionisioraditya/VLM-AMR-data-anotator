"""Modern Theme Stylesheets (Dark and Light Modes) for AMR VLM Dataset Maker."""

DARK_THEME = """
/* Global Window & Base */
QMainWindow, QWidget#centralWidget {
    background-color: #0f172a;
    color: #f8fafc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}

/* Cards and Panels */
QFrame#panelCard, QFrame#card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
}

QGroupBox {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    margin-top: 24px;
    font-weight: 600;
    color: #cbd5e1;
    padding-top: 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #38bdf8;
}

/* Labels */
QLabel {
    color: #e2e8f0;
}
QLabel#titleLabel {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
}
QLabel#subtitleLabel {
    font-size: 12px;
    color: #94a3b8;
}
QLabel#badgeLabel {
    background-color: #334155;
    color: #38bdf8;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}

/* Push Buttons */
QPushButton {
    background-color: #334155;
    color: #f8fafc;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #475569;
    border-color: #64748b;
}
QPushButton:pressed {
    background-color: #1e293b;
}
QPushButton:disabled {
    background-color: #1e293b;
    color: #64748b;
    border-color: #334155;
}

/* Button Variants */
QPushButton#primaryBtn {
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #3b82f6;
}
QPushButton#primaryBtn:hover {
    background-color: #1d4ed8;
    border-color: #60a5fa;
}
QPushButton#primaryBtn:pressed {
    background-color: #1e40af;
}

QPushButton#successBtn {
    background-color: #059669;
    color: #ffffff;
    border: 1px solid #10b981;
}
QPushButton#successBtn:hover {
    background-color: #047857;
    border-color: #34d399;
}

QPushButton#dangerBtn {
    background-color: #dc2626;
    color: #ffffff;
    border: 1px solid #ef4444;
}
QPushButton#dangerBtn:hover {
    background-color: #b91c1c;
    border-color: #f87171;
}

QPushButton#aiBtn {
    background-color: #7c3aed;
    color: #ffffff;
    border: 1px solid #8b5cf6;
}
QPushButton#aiBtn:hover {
    background-color: #6d28d9;
    border-color: #a78bfa;
}

/* Inputs & Form Controls */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0f172a;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: #2563eb;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: 1.5px solid #3b82f6;
    background-color: #111827;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #334155;
}
QComboBox QAbstractItemView {
    background-color: #1e293b;
    color: #f8fafc;
    border: 1px solid #334155;
    selection-background-color: #2563eb;
}

/* Tab Navigation */
QTabWidget::pane {
    border: 1px solid #334155;
    border-radius: 8px;
    background-color: #0f172a;
    top: -1px;
}
QTabBar::tab {
    background-color: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 9px 18px;
    font-weight: 600;
    margin-right: 4px;
}
QTabBar::tab:selected {
    background-color: #0f172a;
    color: #38bdf8;
    border-top: 2px solid #38bdf8;
    border-bottom: 1px solid #0f172a;
}
QTabBar::tab:hover:!selected {
    background-color: #334155;
    color: #f8fafc;
}

/* List Widgets */
QListWidget {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 4px;
}
QListWidget::item {
    color: #e2e8f0;
    padding: 8px 10px;
    border-radius: 6px;
    margin-bottom: 2px;
}
QListWidget::item:hover {
    background-color: #1e293b;
}
QListWidget::item:selected {
    background-color: #2563eb;
    color: #ffffff;
}

/* Progress Bar */
QProgressBar {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    text-align: center;
    color: #f8fafc;
    font-weight: 600;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 5px;
}

/* Scroll Bars */
QScrollBar:vertical {
    background-color: #0f172a;
    width: 10px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background-color: #334155;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background-color: #475569;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Status Bar */
QStatusBar {
    background-color: #1e293b;
    color: #94a3b8;
    border-top: 1px solid #334155;
}
"""

LIGHT_THEME = """
/* Global Window & Base */
QMainWindow, QWidget#centralWidget {
    background-color: #f8fafc;
    color: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}

/* Cards and Panels */
QFrame#panelCard, QFrame#card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
}

QGroupBox {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin-top: 24px;
    font-weight: 600;
    color: #334155;
    padding-top: 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #0284c7;
}

/* Labels */
QLabel {
    color: #1e293b;
}
QLabel#titleLabel {
    font-size: 18px;
    font-weight: 700;
    color: #0f172a;
}
QLabel#subtitleLabel {
    font-size: 12px;
    color: #64748b;
}
QLabel#badgeLabel {
    background-color: #e0f2fe;
    color: #0284c7;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}

/* Push Buttons */
QPushButton {
    background-color: #f1f5f9;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #e2e8f0;
    border-color: #94a3b8;
}
QPushButton:pressed {
    background-color: #cbd5e1;
}
QPushButton:disabled {
    background-color: #f1f5f9;
    color: #94a3b8;
    border-color: #e2e8f0;
}

/* Button Variants */
QPushButton#primaryBtn {
    background-color: #2563eb;
    color: #ffffff;
    border: 1px solid #1d4ed8;
}
QPushButton#primaryBtn:hover {
    background-color: #1d4ed8;
    border-color: #1e40af;
}
QPushButton#primaryBtn:pressed {
    background-color: #1e3a8a;
}

QPushButton#successBtn {
    background-color: #10b981;
    color: #ffffff;
    border: 1px solid #059669;
}
QPushButton#successBtn:hover {
    background-color: #059669;
}

QPushButton#dangerBtn {
    background-color: #ef4444;
    color: #ffffff;
    border: 1px solid #dc2626;
}
QPushButton#dangerBtn:hover {
    background-color: #dc2626;
}

QPushButton#aiBtn {
    background-color: #8b5cf6;
    color: #ffffff;
    border: 1px solid #7c3aed;
}
QPushButton#aiBtn:hover {
    background-color: #7c3aed;
}

/* Inputs & Form Controls */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #ffffff;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: #3b82f6;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: 1.5px solid #2563eb;
    background-color: #ffffff;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #cbd5e1;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    selection-background-color: #e0f2fe;
    selection-color: #0369a1;
}

/* Tab Navigation */
QTabWidget::pane {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background-color: #ffffff;
    top: -1px;
}
QTabBar::tab {
    background-color: #f1f5f9;
    color: #64748b;
    border: 1px solid #e2e8f0;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 9px 18px;
    font-weight: 600;
    margin-right: 4px;
}
QTabBar::tab:selected {
    background-color: #ffffff;
    color: #0284c7;
    border-top: 2px solid #0284c7;
    border-bottom: 1px solid #ffffff;
}
QTabBar::tab:hover:!selected {
    background-color: #e2e8f0;
    color: #0f172a;
}

/* List Widgets */
QListWidget {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 4px;
}
QListWidget::item {
    color: #1e293b;
    padding: 8px 10px;
    border-radius: 6px;
    margin-bottom: 2px;
}
QListWidget::item:hover {
    background-color: #f1f5f9;
}
QListWidget::item:selected {
    background-color: #e0f2fe;
    color: #0284c7;
    font-weight: 600;
}

/* Progress Bar */
QProgressBar {
    background-color: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    text-align: center;
    color: #0f172a;
    font-weight: 600;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 5px;
}

/* Scroll Bars */
QScrollBar:vertical {
    background-color: #f8fafc;
    width: 10px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background-color: #cbd5e1;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background-color: #94a3b8;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Status Bar */
QStatusBar {
    background-color: #f1f5f9;
    color: #64748b;
    border-top: 1px solid #e2e8f0;
}
"""
