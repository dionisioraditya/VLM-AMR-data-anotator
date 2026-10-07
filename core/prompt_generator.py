import os
import json
import random
from typing import List, Dict, Any, Optional, Set

class TemplatePromptGenerator:
    """Generates varied VLM prompt samples from template_prompt.txt for object grounding and frontier exploration."""

    DEFAULT_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
    DEFAULT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "template_prompt.txt")

    @classmethod
    def get_default_template_path_for_split(cls, split: str = "train") -> str:
        """Get the default master template file path for train, val, or test split."""
        split_clean = (split or "train").lower().strip()
        candidate = os.path.join(cls.DEFAULT_TEMPLATES_DIR, f"{split_clean}_template.txt")
        if os.path.isfile(candidate):
            return candidate
        return cls.DEFAULT_TEMPLATE_PATH

    def __init__(self, template_file_path: Optional[str] = None, split: str = "train"):
        self.split = (split or "train").lower().strip()
        if template_file_path:
            self.template_file_path = template_file_path
        else:
            self.template_file_path = self.get_default_template_path_for_split(self.split)
        self.id_templates: List[str] = []
        self.en_templates: List[str] = []
        self.id_templates_set: Set[str] = set()
        self.en_templates_set: Set[str] = set()
        self._load_templates()

    def _load_templates(self):
        """Parse id and en templates from file."""
        self.id_templates = []
        self.en_templates = []
        if not os.path.isfile(self.template_file_path):
            return

        with open(self.template_file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f]

        current_lang = None
        for line in lines:
            line_clean = line.strip()
            if line_clean.lower() in ("id:", "[id]"):
                current_lang = "id"
                continue
            elif line_clean.lower() in ("en:", "[en]"):
                current_lang = "en"
                continue

            if not line_clean:
                continue

            # Skip category headers like '1. Instruksi...' or '2. Location...'
            if line_clean[0].isdigit() and ("." in line_clean[:4] or ")" in line_clean[:4]):
                continue

            if "X" in line_clean:
                if current_lang == "id":
                    self.id_templates.append(line_clean)
                elif current_lang == "en":
                    self.en_templates.append(line_clean)
                else:
                    self.id_templates.append(line_clean)

        self.id_templates_set = set(self.id_templates)
        self.en_templates_set = set(self.en_templates)

    def add_template(self, text: str, lang: str = "id") -> bool:
        """Add a new template string containing 'X'."""
        text = text.strip()
        if not text or "X" not in text:
            return False
        if lang == "id":
            if text not in self.id_templates_set:
                self.id_templates.append(text)
                self.id_templates_set.add(text)
                return True
        elif lang == "en":
            if text not in self.en_templates_set:
                self.en_templates.append(text)
                self.en_templates_set.add(text)
                return True
        return False

    def remove_template(self, text: str, lang: str = "id") -> bool:
        """Remove a template string."""
        text = text.strip()
        if lang == "id" and text in self.id_templates_set:
            self.id_templates.remove(text)
            self.id_templates_set.remove(text)
            return True
        elif lang == "en" and text in self.en_templates_set:
            self.en_templates.remove(text)
            self.en_templates_set.remove(text)
            return True
        return False

    def edit_template(self, old_text: str, new_text: str, lang: str = "id") -> bool:
        """Update an existing template string."""
        old_text = old_text.strip()
        new_text = new_text.strip()
        if not new_text or "X" not in new_text:
            return False
        if lang == "id" and old_text in self.id_templates:
            idx = self.id_templates.index(old_text)
            self.id_templates[idx] = new_text
            self.id_templates_set.remove(old_text)
            self.id_templates_set.add(new_text)
            return True
        elif lang == "en" and old_text in self.en_templates:
            idx = self.en_templates.index(old_text)
            self.en_templates[idx] = new_text
            self.en_templates_set.remove(old_text)
            self.en_templates_set.add(new_text)
            return True
        return False

    def save_templates_to_file(self, target_path: Optional[str] = None):
        """Save in-memory templates to file."""
        dest = target_path or self.template_file_path
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write("id:\n")
            for tpl in self.id_templates:
                f.write(f"{tpl}\n")
            f.write("\nen:\n")
            for tpl in self.en_templates:
                f.write(f"{tpl}\n")

    def reload(self):
        """Reload templates from disk."""
        self._load_templates()

    def get_template_language(self, template: str) -> str:
        """Determine if a template is Indonesian ('id') or English ('en')."""
        if template in self.id_templates_set:
            return "id"
        if template in self.en_templates_set:
            return "en"
        return "id"

    def get_templates(self, language: str = "id") -> List[str]:
        """
        Get template list for specified language.
        language: 'id', 'en', or 'both'
        """
        if language == "id":
            return list(self.id_templates)
        elif language == "en":
            return list(self.en_templates)
        elif language == "both":
            return list(self.id_templates) + list(self.en_templates)
        return list(self.id_templates)

    @staticmethod
    def calculate_dynamic_frontier_score(box_2d: List[int]) -> float:
        """
        Calculate dynamic frontier exploration score based on 2D bounding box size.
        Box format: [ymin, xmin, ymax, xmax] normalized to 0-1000.

        Rationale for AMR:
        - Small box (far away) -> Depth camera pointcloud cannot reach -> High frontier score (0.75 - 0.95).
        - Large box (close up) -> Depth camera pointcloud has precise metric 3D -> Low frontier score (0.15 - 0.35).
        """
        if not box_2d or len(box_2d) < 4:
            return 0.50
        ymin, xmin, ymax, xmax = box_2d[:4]
        h = max(0, min(1000, ymax) - max(0, min(1000, ymin)))
        w = max(0, min(1000, xmax) - max(0, min(1000, xmin)))
        area_norm = (h * w) / 1_000_000.0  # Fraction between 0.0 and 1.0

        # S = linear scale factor proportional to 1/distance
        s = area_norm ** 0.5

        # Linear decay from 0.95 (tiny box, far away) down to 0.15 (large box, close up)
        score = 0.95 - (1.33 * s)
        score = max(0.15, min(0.95, score))
        return round(score, 2)

    def generate_frame_prompts(
        self,
        boxes: List[Dict[str, Any]],
        project_classes: List[str],
        language: str = "id",
        max_templates_per_class: Optional[int] = None,
        frontier_score: float = 0.85,
        existing_prompts: Optional[List[Dict[str, Any]]] = None,
        shuffle: bool = False,
        class_synonyms: Optional[Dict[str, Dict[str, List[str]]]] = None,
        frontier_mode: str = "dynamic",
        positive_frontier_score: float = 0.85,
    ) -> List[Dict[str, Any]]:
        """
        Generate prompt variations for a single frame.

        - For detected boxes (positive):
          {"target_detected": true, "label": label, "bounding_box": box_2d, "frontier_score": score}
          (score is calculated dynamically from box size if frontier_mode == 'dynamic', or positive_frontier_score if 'static')
        - For missing project classes (negative / frontier exploration):
          {"target_detected": false, "label": null, "bounding_box": null, "frontier_score": frontier_score}

        Args:
            boxes: Detected bounding boxes in the frame
            project_classes: List of canonical class names in project
            language: 'id', 'en', or 'both'
            max_templates_per_class: Max templates to use per class
            frontier_score: Score to assign for non-present target classes (negative)
            existing_prompts: Previously generated prompts to avoid duplicates
            shuffle: Whether to shuffle template choices
            class_synonyms: Optional mapping of class label -> {'id': [...], 'en': [...]}
            frontier_mode: 'dynamic' (auto from BBOX) or 'static' (fixed score)
            positive_frontier_score: Score to assign if frontier_mode is 'static'
        Returns:
            List of prompt dicts: [{"id": "p_...", "user": "...", "assistant": "..."}, ...]
        """
        templates = self.get_templates(language)
        if not templates:
            return []

        existing_prompts = existing_prompts or []
        existing_user_texts: Set[str] = {p.get("user", "") for p in existing_prompts}
        result_prompts: List[Dict[str, Any]] = []

        # Helper to get candidate synonyms for a class label and template language
        def get_candidates(label_name: str, tpl_lang: str) -> List[str]:
            if class_synonyms:
                l_key = label_name.strip().lower()
                for k, v in class_synonyms.items():
                    if k.strip().lower() == l_key and isinstance(v, dict):
                        syn_list = v.get(tpl_lang, [])
                        clean_syns = [s.strip() for s in syn_list if s and s.strip()]
                        if clean_syns:
                            return list(clean_syns)
            return [label_name]

        # Track labels present in boxes (case-insensitive)
        boxes = boxes or []
        present_labels_map: Dict[str, List[Dict[str, Any]]] = {}
        for b in boxes:
            lbl = str(b.get("label", "object")).strip()
            lbl_key = lbl.lower()
            if lbl_key not in present_labels_map:
                present_labels_map[lbl_key] = []
            present_labels_map[lbl_key].append(b)

        # 1. POSITIVE PROMPTS (For each detected box)
        for lbl_key, instance_boxes in present_labels_map.items():
            num_instances = len(instance_boxes)
            # Select templates for this class
            class_templates = list(templates)
            if shuffle:
                random.shuffle(class_templates)

            if max_templates_per_class and max_templates_per_class > 0:
                needed = min(max_templates_per_class * num_instances, len(class_templates))
                class_templates = class_templates[:needed]

            # Distribute templates across instances of this label so each box gets unique prompt variations
            for i, b in enumerate(instance_boxes):
                lbl = str(b.get("label", "object")).strip()
                b_2d = b.get("box_2d", [0, 0, 0, 0])

                if frontier_mode == "dynamic":
                    pos_score = self.calculate_dynamic_frontier_score(b_2d)
                elif frontier_mode == "static":
                    pos_score = round(float(positive_frontier_score), 2)
                else:  # "null", "none", or without frontier score
                    pos_score = None

                payload = {
                    "target_detected": True,
                    "label": lbl,
                    "bounding_box": b_2d,
                    "frontier_score": pos_score,
                }
                assistant_json = json.dumps(payload, ensure_ascii=False)

                # Slice templates for this specific instance
                instance_templates = [t for idx, t in enumerate(class_templates) if idx % num_instances == i]
                if not instance_templates and class_templates:
                    instance_templates = [class_templates[i % len(class_templates)]]

                for t_idx, tpl in enumerate(instance_templates):
                    tpl_lang = self.get_template_language(tpl)
                    candidates = get_candidates(lbl, tpl_lang)
                    if shuffle and len(candidates) > 1:
                        candidates = list(candidates)
                        random.shuffle(candidates)

                    # Cycle candidate synonyms starting from t_idx
                    start_idx = t_idx % len(candidates)
                    ordered_candidates = candidates[start_idx:] + candidates[:start_idx]

                    chosen_text = None
                    for syn in ordered_candidates:
                        u_text = tpl.replace("X", syn)
                        if u_text not in existing_user_texts:
                            chosen_text = u_text
                            break

                    if chosen_text:
                        prompt_id = f"p_{len(existing_prompts) + len(result_prompts) + 1}"
                        result_prompts.append({
                            "id": prompt_id,
                            "user": chosen_text,
                            "assistant": assistant_json,
                        })
                        existing_user_texts.add(chosen_text)

        # 2. NEGATIVE / FRONTIER EXPLORATION PROMPTS (For project classes not detected in this frame)
        missing_classes = [c for c in project_classes if c.strip().lower() not in present_labels_map]
        neg_payload = {
            "target_detected": False,
            "label": None,
            "bounding_box": None,
            "frontier_score": round(float(frontier_score), 2),
        }
        neg_assistant_json = json.dumps(neg_payload, ensure_ascii=False)

        for missing_c in missing_classes:
            c_label = missing_c.strip()
            class_templates = list(templates)
            if shuffle:
                random.shuffle(class_templates)

            if max_templates_per_class and max_templates_per_class > 0:
                class_templates = class_templates[:max_templates_per_class]

            for t_idx, tpl in enumerate(class_templates):
                tpl_lang = self.get_template_language(tpl)
                candidates = get_candidates(c_label, tpl_lang)
                if shuffle and len(candidates) > 1:
                    candidates = list(candidates)
                    random.shuffle(candidates)

                start_idx = t_idx % len(candidates)
                ordered_candidates = candidates[start_idx:] + candidates[:start_idx]

                chosen_text = None
                for syn in ordered_candidates:
                    u_text = tpl.replace("X", syn)
                    if u_text not in existing_user_texts:
                        chosen_text = u_text
                        break

                if chosen_text:
                    prompt_id = f"p_{len(existing_prompts) + len(result_prompts) + 1}"
                    result_prompts.append({
                        "id": prompt_id,
                        "user": chosen_text,
                        "assistant": neg_assistant_json,
                    })
                    existing_user_texts.add(chosen_text)

        return result_prompts
