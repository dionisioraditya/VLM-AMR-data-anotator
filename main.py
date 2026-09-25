#!/usr/bin/env python3
"""
AMR VLM Dataset Maker
Desktop Application for Vision-Language Model AMR Object Grounding Dataset Creation.
"""

import sys
import os
import argparse

# Ensure current project directory is in python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Preload libxcb-cursor if available in ~/.local/lib to prevent Qt xcb plugin load failure
try:
    import ctypes
    local_xcb_cursor = os.path.expanduser("~/.local/lib/libxcb-cursor.so.0")
    if os.path.isfile(local_xcb_cursor):
        ctypes.CDLL(local_xcb_cursor, mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from ui.app_window import MainWindow

def main():
    parser = argparse.ArgumentParser(description="AMR VLM Dataset Maker")
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Path ke folder project dataset (opsional).",
    )
    args = parser.parse_args()

    # Enable High DPI pixmaps
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("AMR VLM Dataset Maker")
    app.setOrganizationName("AMR-Robotics")

    # Launch Main Window
    window = MainWindow(initial_project_dir=args.project)
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
