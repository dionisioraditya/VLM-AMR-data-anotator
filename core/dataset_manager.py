import os
import json
import shutil
import random
from typing import List, Dict, Any, Optional, Tuple
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

    DEFAULT_CLASS_SYNONYMS = {
        "trash_bin": {
            "id": ["tempat sampah", "tong sampah", "keranjang sampah", "bak sampah"],
            "en": ["trash bin", "trash can", "garbage can", "rubbish bin", "dustbin", "waste basket"]
        },
        "door": {
            "id": ["pintu", "pintu ruangan", "pintu masuk", "pintu keluar"],
            "en": ["door", "doorway", "entrance", "exit door"]
        },
        "cardboard box": {
            "id": ["kardus", "kotak kardus", "dus", "box"],
            "en": ["cardboard box", "carton", "box", "package"]
        },
        "dispenser": {
            "id": ["dispenser", "dispenser air", "mesin air minum"],
            "en": ["water dispenser", "water cooler", "dispenser"]
        },
        "fire extinguisher": {
            "id": ["alat pemadam api", "apar", "tabung pemadam"],
            "en": ["fire extinguisher", "extinguisher"]
        },
        "person": {
            "id": ["orang", "manusia", "pejalan kaki", "seseorang"],
            "en": ["person", "human", "pedestrian", "someone"]
        },
        "pallet": {
            "id": ["palet", "palet kayu", "tatakan palet"],
            "en": ["pallet", "wooden pallet", "skid"]
        },
        "chair": {
            "id": ["kursi", "tempat duduk", "bangku"],
            "en": ["chair", "seat", "stool"]
        },
    }

    def __init__(self, project_dir: Optional[str] = None):
        self.project_dir: Optional[str] = None
        self.frames_dir: Optional[str] = None
        self.active_frames_dir: Optional[str] = None
        self.annotations_dir: Optional[str] = None
        self.active_annotations_dir: Optional[str] = None
        self.config_path: Optional[str] = None
        self.classes: List[str] = list(self.DEFAULT_CLASSES)
        self.class_synonyms: Dict[str, Dict[str, List[str]]] = {
            k: {"id": list(v.get("id", [])), "en": list(v.get("en", []))}
            for k, v in self.DEFAULT_CLASS_SYNONYMS.items()
        }
        self.DEFAULT_GENERATOR_SETTINGS: Dict[str, Any] = {
            "language": "both",
            "use_limit": False,
            "max_per_class": None,
            "limit_spin_value": 10,
            "frontier_score": 0.85,
            "scope": "current",
            "clear_existing": False,
        }
        self.generator_settings: Dict[str, Any] = dict(self.DEFAULT_GENERATOR_SETTINGS)
        self.splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        self.templates_dir: Optional[str] = None

        if project_dir:
            self.set_project_dir(project_dir)

    def set_project_dir(self, project_dir: str):
        """Set project root directory and ensure folders exist."""
        self.project_dir = project_dir
        self.frames_dir = os.path.join(project_dir, "frames")
        self.active_frames_dir = self.frames_dir
        self.annotations_dir = os.path.join(project_dir, "annotations")
        self.active_annotations_dir = self.annotations_dir
        self.templates_dir = os.path.join(project_dir, "templates")
        self.config_path = os.path.join(project_dir, "project_config.json")

        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
        self._ensure_project_templates()

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
                    loaded_syn = cfg.get("class_synonyms", {})
                    # Seed with default synonyms for standard AMR classes
                    self.class_synonyms = {
                        k: {"id": list(v.get("id", [])), "en": list(v.get("en", []))}
                        for k, v in self.DEFAULT_CLASS_SYNONYMS.items()
                    }
                    # Overlay custom/saved synonyms
                    for k, v in loaded_syn.items():
                        if isinstance(v, dict):
                            self.class_synonyms[k] = {
                                "id": list(v.get("id", [])),
                                "en": list(v.get("en", []))
                            }
                    # Load saved generator settings
                    self.generator_settings = dict(self.DEFAULT_GENERATOR_SETTINGS)
                    self.generator_settings.update(cfg.get("generator_settings", {}))
                    # Load saved splits
                    loaded_splits = cfg.get("splits", {})
                    self.splits = {
                        "train": list(loaded_splits.get("train", [])),
                        "val": list(loaded_splits.get("val", [])),
                        "test": list(loaded_splits.get("test", [])),
                    }
            except Exception:
                self.classes = list(self.DEFAULT_CLASSES)
                self.splits = {"train": [], "val": [], "test": []}
        else:
            self._save_config()

    def _ensure_project_templates(self):
        """Ensure project has copies of train, val, and test templates."""
        if not self.templates_dir:
            return
        os.makedirs(self.templates_dir, exist_ok=True)
        master_dir = os.path.join(os.path.dirname(__file__), "templates")
        for split in ("train", "val", "test"):
            p_tpl = os.path.join(self.templates_dir, f"{split}_template.txt")
            if not os.path.isfile(p_tpl):
                m_tpl = os.path.join(master_dir, f"{split}_template.txt")
                if os.path.isfile(m_tpl):
                    shutil.copy2(m_tpl, p_tpl)
                else:
                    legacy = os.path.join(os.path.dirname(__file__), "template_prompt.txt")
                    if os.path.isfile(legacy):
                        shutil.copy2(legacy, p_tpl)

    def get_template_path_for_split(self, split: str = "train") -> str:
        """Get path to project template file for a given split (train/val/test)."""
        split_clean = (split or "train").lower().strip()
        if self.templates_dir:
            p_tpl = os.path.join(self.templates_dir, f"{split_clean}_template.txt")
            if os.path.isfile(p_tpl):
                return p_tpl
            self._ensure_project_templates()
            if os.path.isfile(p_tpl):
                return p_tpl

        master_dir = os.path.join(os.path.dirname(__file__), "templates")
        m_tpl = os.path.join(master_dir, f"{split_clean}_template.txt")
        if os.path.isfile(m_tpl):
            return m_tpl
        return os.path.join(os.path.dirname(__file__), "template_prompt.txt")

    def _save_config(self):
        """Save project configuration."""
        if not self.config_path:
            return
        cfg = {
            "classes": self.classes,
            "class_synonyms": self.class_synonyms,
            "generator_settings": self.generator_settings,
            "splits": self.splits,
            "version": "1.0",
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

    def get_generator_settings(self) -> Dict[str, Any]:
        """Return the last saved prompt generator settings."""
        return dict(self.generator_settings)

    def save_generator_settings(self, settings: Dict[str, Any]):
        """Persist user-selected prompt generator settings for subsequent runs."""
        self.generator_settings.update(settings)
        self._save_config()

    # ---------------- Dataset Split Management (Train, Val, Test) ----------------
    def get_splits(self) -> Dict[str, List[str]]:
        """Return synchronized splits mapping of split -> image list."""
        all_imgs = self.get_image_list()
        all_set = set(all_imgs)

        train_set = [img for img in self.splits.get("train", []) if img in all_set]
        val_set = [img for img in self.splits.get("val", []) if img in all_set]
        test_set = [img for img in self.splits.get("test", []) if img in all_set]

        assigned = set(train_set) | set(val_set) | set(test_set)
        unassigned = [img for img in all_imgs if img not in assigned]
        train_set.extend(unassigned)

        self.splits = {
            "train": train_set,
            "val": val_set,
            "test": test_set,
        }
        return self.splits

    def set_splits(self, splits: Dict[str, List[str]]):
        """Persist user-curated splits."""
        all_set = set(self.get_image_list())
        self.splits = {
            "train": [img for img in splits.get("train", []) if img in all_set],
            "val": [img for img in splits.get("val", []) if img in all_set],
            "test": [img for img in splits.get("test", []) if img in all_set],
        }
        self._save_config()

    def get_images_for_split(self, split: str) -> List[str]:
        """Get image filenames belonging to a specific split."""
        splits = self.get_splits()
        return list(splits.get((split or "train").lower().strip(), []))

    def get_split_for_image(self, image_name: str) -> str:
        """Find which split an image belongs to ('train', 'val', or 'test')."""
        splits = self.get_splits()
        for s_name in ("train", "val", "test"):
            if image_name in splits[s_name]:
                return s_name
        return "train"

    def move_image_to_split(self, image_name: str, target_split: str) -> bool:
        """Move an image to target_split."""
        target = (target_split or "").lower().strip()
        if target not in ("train", "val", "test"):
            return False
        splits = self.get_splits()
        for s in ("train", "val", "test"):
            if image_name in splits[s]:
                splits[s].remove(image_name)
        splits[target].append(image_name)
        self.set_splits(splits)
        return True

    def auto_split(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = True,
        seed: int = 42,
        **kwargs,
    ) -> Dict[str, List[str]]:
        """
        Automatically split all project images into train, val, and test.
        If stratify is True, balances frames with bounding boxes vs empty/negative frames.
        """
        if "train_pct" in kwargs:
            val_p = kwargs.get("train_pct", 70)
            train_ratio = val_p / 100.0 if val_p > 1.0 else float(val_p)
        if "val_pct" in kwargs:
            val_p = kwargs.get("val_pct", 15)
            val_ratio = val_p / 100.0 if val_p > 1.0 else float(val_p)
        if "test_pct" in kwargs:
            val_p = kwargs.get("test_pct", 15)
            test_ratio = val_p / 100.0 if val_p > 1.0 else float(val_p)
        if "random_seed" in kwargs:
            seed = kwargs["random_seed"]
        all_imgs = self.get_image_list()
        if not all_imgs:
            self.splits = {"train": [], "val": [], "test": []}
            self._save_config()
            return self.splits

        total_r = train_ratio + val_ratio + test_ratio
        if total_r <= 0:
            train_ratio, val_ratio, test_ratio = 0.70, 0.15, 0.15
            total_r = 1.0

        r_train = train_ratio / total_r
        r_val = val_ratio / total_r

        import random
        rng = random.Random(seed)

        def partition(items: List[str]) -> Tuple[List[str], List[str], List[str]]:
            shuffled = list(items)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_tr = int(round(n * r_train))
            n_va = int(round(n * r_val))
            # Ensure within bounds
            n_tr = min(n, max(0, n_tr))
            n_va = min(n - n_tr, max(0, n_va))
            tr = shuffled[:n_tr]
            va = shuffled[n_tr:n_tr + n_va]
            te = shuffled[n_tr + n_va:]
            return tr, va, te

        if stratify:
            pos_imgs = [img for img in all_imgs if self.is_annotated(img)]
            neg_imgs = [img for img in all_imgs if not self.is_annotated(img)]
            p_tr, p_va, p_te = partition(pos_imgs)
            n_tr, n_va, n_te = partition(neg_imgs)
            new_train = p_tr + n_tr
            new_val = p_va + n_va
            new_test = p_te + n_te
        else:
            new_train, new_val, new_test = partition(all_imgs)

        self.splits = {
            "train": new_train,
            "val": new_val,
            "test": new_test,
        }
        self._save_config()
        return self.splits

    def add_class(self, label: str):
        """Add a new target class."""
        label = label.strip()
        if label and label not in self.classes:
            self.classes.append(label)
            if label not in self.class_synonyms:
                self.class_synonyms[label] = {"id": [label], "en": [label]}
            self._save_config()

    def remove_class(self, label: str):
        """Remove a class."""
        if label in self.classes:
            self.classes.remove(label)
            # Remove from class_synonyms as well
            for k in list(self.class_synonyms.keys()):
                if k.lower() == label.lower():
                    self.class_synonyms.pop(k, None)
                    break
            self._save_config()

    def rename_class(self, old_label: str, new_label: str, update_annotations: bool = True) -> int:
        """
        Renames a class in project config, and optionally updates all bounding boxes
        across all annotations in the project.

        Returns:
            Number of bounding boxes updated.
        """
        old_clean = old_label.strip()
        new_clean = new_label.strip()

        if not old_clean or not new_clean or old_clean == new_clean:
            return 0

        # Update in project classes list
        if old_clean in self.classes:
            idx = self.classes.index(old_clean)
            if new_clean in self.classes:
                # new_clean already exists, remove old_clean to avoid duplicates
                self.classes.pop(idx)
            else:
                self.classes[idx] = new_clean
        elif new_clean not in self.classes:
            self.classes.append(new_clean)

        # Update in class_synonyms mapping
        for k in list(self.class_synonyms.keys()):
            if k.lower() == old_clean.lower():
                val = self.class_synonyms.pop(k)
                self.class_synonyms[new_clean] = val
                break

        self._save_config()

        updated_count = 0
        if update_annotations and self.annotations_dir and os.path.isdir(self.annotations_dir):
            for root, _, files in os.walk(self.annotations_dir):
                for f in files:
                    if f.endswith(".json") and f != "config.json":
                        anno_path = os.path.join(root, f)
                        try:
                            with open(anno_path, "r", encoding="utf-8") as af:
                                data = json.load(af)

                            changed = False
                            for b in data.get("boxes", []):
                                if b.get("label") == old_clean:
                                    b["label"] = new_clean
                                    changed = True
                                    updated_count += 1

                            if changed:
                                with open(anno_path, "w", encoding="utf-8") as af:
                                    json.dump(data, af, indent=2, ensure_ascii=False)
                        except Exception:
                            pass

        return updated_count

    def get_class_synonyms(self, label: str, lang: str = "id") -> List[str]:
        """Get synonyms list for a class in 'id' or 'en'."""
        lbl_clean = label.strip().lower()
        lang_clean = lang.strip().lower()
        for k, v in self.class_synonyms.items():
            if k.lower() == lbl_clean:
                return list(v.get(lang_clean, []))
        return []

    def set_class_synonyms(self, label: str, lang: str, synonyms: List[str]):
        """Set synonyms list for a class in 'id' or 'en'."""
        lbl_clean = label.strip()
        lang_clean = lang.strip().lower()
        if not lbl_clean or lang_clean not in ("id", "en"):
            return

        match_key = None
        for k in self.class_synonyms.keys():
            if k.lower() == lbl_clean.lower():
                match_key = k
                break
        if not match_key:
            match_key = lbl_clean
            self.class_synonyms[match_key] = {"id": [], "en": []}

        cleaned = [s.strip() for s in synonyms if s.strip()]
        self.class_synonyms[match_key][lang_clean] = cleaned
        self._save_config()

    def get_all_synonyms(self) -> Dict[str, Dict[str, List[str]]]:
        """Return full copy of all class synonyms."""
        return {
            k: {"id": list(v.get("id", [])), "en": list(v.get("en", []))}
            for k, v in self.class_synonyms.items()
        }

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

        # Remove from splits
        for s in ("train", "val", "test"):
            if image_name in self.splits.get(s, []):
                self.splits[s].remove(image_name)
        self._save_config()

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
