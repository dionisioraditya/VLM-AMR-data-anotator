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
