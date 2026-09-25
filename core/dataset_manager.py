import os
import json
from typing import List, Dict, Any, Optional
from PIL import Image

class DatasetManager:
    """Manages project data, frames, annotations, and prompt variations."""

    DEFAULT_CLASSES = [
        "pallet",
        "wooden_pallet",
        "forklift",
        "human",
        "worker",
        "charging_dock",
        "box",
        "cone",
        "obstacle",
        "amr_robot",
    ]

    def __init__(self, project_dir: Optional[str] = None):
        self.project_dir: Optional[str] = None
        self.frames_dir: Optional[str] = None
        self.active_frames_dir: Optional[str] = None
        self.annotations_dir: Optional[str] = None
        self.active_annotations_dir: Optional[str] = None
        self.config_path: Optional[str] = None
        self.classes: List[str] = list(self.DEFAULT_CLASSES)

        if project_dir:
            self.set_project_dir(project_dir)

    def set_project_dir(self, project_dir: str):
        """Set project root directory and ensure folders exist."""
        self.project_dir = project_dir
        self.frames_dir = os.path.join(project_dir, "frames")
        self.active_frames_dir = self.frames_dir
        self.annotations_dir = os.path.join(project_dir, "annotations")
        self.active_annotations_dir = self.annotations_dir
        self.config_path = os.path.join(project_dir, "project_config.json")

        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)

        self._load_config()

        # Auto-detect if frames_dir has subfolders with images (e.g. frames/dispenser)
        subfolders = self.get_subfolders()
        if subfolders:
            # If root frames has no direct images, auto-select the first subfolder with images
            root_imgs = self._scan_dir_images(self.frames_dir)
            if not root_imgs:
                for sub in subfolders:
                    if sub["count"] > 0:
                        self.set_active_frames_dir(sub["path"])
                        break

    def set_active_frames_dir(self, directory: str):
        """Set currently active directory for images and sync corresponding annotations dir."""
        self.active_frames_dir = os.path.abspath(directory)
        os.makedirs(self.active_frames_dir, exist_ok=True)

        # Mirror structure to annotations/
        if self.project_dir and self.active_frames_dir.startswith(self.frames_dir):
            rel_path = os.path.relpath(self.active_frames_dir, self.frames_dir)
            if rel_path != ".":
                self.active_annotations_dir = os.path.join(self.annotations_dir, rel_path)
            else:
                self.active_annotations_dir = self.annotations_dir
        else:
            folder_name = os.path.basename(self.active_frames_dir)
            self.active_annotations_dir = os.path.join(self.annotations_dir, folder_name)

        os.makedirs(self.active_annotations_dir, exist_ok=True)

    def get_subfolders(self) -> List[Dict[str, Any]]:
        """Return list of available subfolders inside frames_dir with image counts."""
        if not self.frames_dir or not os.path.isdir(self.frames_dir):
            return []

        result = []
        # Include root if it has images or if no subfolders exist
        root_count = len(self._scan_dir_images(self.frames_dir))
        result.append({
            "name": "(Root) frames",
            "rel_path": "",
            "path": self.frames_dir,
            "count": root_count,
        })

        try:
            entries = sorted(os.listdir(self.frames_dir))
            for e in entries:
                full_p = os.path.join(self.frames_dir, e)
                if os.path.isdir(full_p):
                    count = len(self._scan_dir_images(full_p))
                    result.append({
                        "name": e,
                        "rel_path": e,
                        "path": full_p,
                        "count": count,
                    })
        except Exception:
            pass

        return result

    def _scan_dir_images(self, directory: str) -> List[str]:
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        if not os.path.isdir(directory):
            return []
        try:
            return [
                f for f in os.listdir(directory)
                if os.path.splitext(f)[1].lower() in valid_exts
            ]
        except Exception:
            return []

    def _load_config(self):
        """Load project configuration (classes, settings)."""
        if self.config_path and os.path.isfile(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    self.classes = cfg.get("classes", list(self.DEFAULT_CLASSES))
            except Exception:
                self.classes = list(self.DEFAULT_CLASSES)
        else:
            self._save_config()

    def _save_config(self):
        """Save project configuration."""
        if not self.config_path:
            return
        cfg = {
            "classes": self.classes,
            "version": "1.0",
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    def add_class(self, label: str):
        """Add a new target class."""
        label = label.strip()
        if label and label not in self.classes:
            self.classes.append(label)
            self._save_config()

    def remove_class(self, label: str):
        """Remove a class."""
        if label in self.classes:
            self.classes.remove(label)
            self._save_config()

    def get_image_list(self) -> List[str]:
        """Return sorted list of image filenames in active_frames_dir."""
        target_dir = self.active_frames_dir or self.frames_dir
        if not target_dir or not os.path.isdir(target_dir):
            return []
        files = self._scan_dir_images(target_dir)
        return sorted(files)

    def get_image_path(self, image_name: str) -> str:
        """Get absolute path to an image file."""
        target_dir = self.active_frames_dir or self.frames_dir
        return os.path.join(target_dir, image_name)

    def get_annotation_path(self, image_name: str) -> str:
        """Get absolute path to annotation file for given image."""
        target_anno = self.active_annotations_dir or self.annotations_dir
        base_name = os.path.splitext(image_name)[0]
        return os.path.join(target_anno, f"{base_name}.json")

    def get_image_size(self, image_name: str) -> tuple[int, int]:
        """Get image (width, height) using PIL without loading full image into memory."""
        path = self.get_image_path(image_name)
        try:
            with Image.open(path) as img:
                return img.size  # (width, height)
        except Exception:
            return (1920, 1080)

    def get_annotation(self, image_name: str) -> Dict[str, Any]:
        """Load annotation for an image, or create a default structure."""
        anno_path = self.get_annotation_path(image_name)
        if os.path.isfile(anno_path):
            try:
                with open(anno_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        w, h = self.get_image_size(image_name)
        return {
            "image_file": image_name,
            "width": w,
            "height": h,
            "boxes": [],     # list of {"id": str, "label": str, "box_2d": [ymin, xmin, ymax, xmax]}
            "prompts": [],   # list of {"id": str, "user": str, "assistant": str}
        }

    def save_annotation(self, image_name: str, data: Dict[str, Any]):
        """Save annotation data for an image."""
        anno_path = self.get_annotation_path(image_name)
        os.makedirs(os.path.dirname(anno_path), exist_ok=True)
        with open(anno_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def delete_image(self, image_name: str) -> bool:
        """Delete an image and its corresponding annotation file."""
        img_path = self.get_image_path(image_name)
        anno_path = self.get_annotation_path(image_name)

        success = True
        if os.path.isfile(img_path):
            try:
                os.remove(img_path)
            except Exception:
                success = False

        if os.path.isfile(anno_path):
            try:
                os.remove(anno_path)
            except Exception:
                pass

        return success

    def is_annotated(self, image_name: str) -> bool:
        """Check if an image has any bounding boxes saved."""
        anno_path = self.get_annotation_path(image_name)
        if not os.path.isfile(anno_path):
            return False
        try:
            with open(anno_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return len(data.get("boxes", [])) > 0
        except Exception:
            return False

    @staticmethod
    def denormalize_box(box_2d: List[int], width: int, height: int) -> List[int]:
        """Convert [ymin, xmin, ymax, xmax] in 0-1000 scale to [x, y, w, h] pixels."""
        ymin, xmin, ymax, xmax = box_2d
        x = int(round((xmin / 1000.0) * width))
        y = int(round((ymin / 1000.0) * height))
        w = int(round(((xmax - xmin) / 1000.0) * width))
        h = int(round(((ymax - ymin) / 1000.0) * height))
        return [max(0, x), max(0, y), max(1, w), max(1, h)]

    @staticmethod
    def normalize_box(pixel_box: List[int], width: int, height: int) -> List[int]:
        """Convert [x, y, w, h] pixels to [ymin, xmin, ymax, xmax] in 0-1000 scale."""
        x, y, w, h = pixel_box
        xmin = int(round((x / float(width)) * 1000.0))
        ymin = int(round((y / float(height)) * 1000.0))
        xmax = int(round(((x + w) / float(width)) * 1000.0))
        ymax = int(round(((y + h) / float(height)) * 1000.0))

        # Clamp between 0 and 1000
        xmin = max(0, min(1000, xmin))
        ymin = max(0, min(1000, ymin))
        xmax = max(0, min(1000, xmax))
        ymax = max(0, min(1000, ymax))

        return [ymin, xmin, ymax, xmax]
