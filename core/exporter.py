import os
import json
import shutil
import random
from typing import Dict, Any, List, Tuple

class DatasetExporter:
    """Exports annotated dataset and prompt variations to standard VLM formats."""

    @staticmethod
    def _normalize_frontier_payload(
        assistant_text: str,
        user_text: str = "",
        boxes: List[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Normalizes assistant output to the structured JSON format:
        Positive: {"target_detected": true, "label": "...", "bounding_box": [...], "frontier_score": null}
        Negative: {"target_detected": false, "label": null, "bounding_box": null, "frontier_score": float}

        Returns:
            Tuple of (payload_dict, is_positive)
        """
        a_clean = assistant_text.strip()
        boxes = boxes or []

        # 1. Check if already valid JSON structure
        try:
            parsed = json.loads(a_clean)
            if isinstance(parsed, dict) and "target_detected" in parsed:
                is_pos = bool(parsed.get("target_detected"))
                # Ensure standard keys exist
                payload = {
                    "target_detected": is_pos,
                    "label": parsed.get("label") if is_pos else None,
                    "bounding_box": parsed.get("bounding_box") if is_pos else None,
                    "frontier_score": parsed.get("frontier_score") if not is_pos else None,
                }
                return payload, is_pos
        except Exception:
            pass

        # 2. Legacy 'null' string
        if a_clean.lower() == "null":
            return {
                "target_detected": False,
                "label": None,
                "bounding_box": None,
                "frontier_score": 0.85,
            }, False

        # 3. Legacy coordinate string: "[ymin, xmin, ymax, xmax]"
        coords = None
        try:
            parsed_coords = json.loads(a_clean)
            if isinstance(parsed_coords, list) and len(parsed_coords) == 4:
                coords = [int(v) for v in parsed_coords]
        except Exception:
            pass

        if coords is not None:
            # Determine label from boxes or extract from quotes in user prompt
            lbl = "object"
            if boxes:
                lbl = boxes[0].get("label", "object")
            elif "'" in user_text:
                parts = user_text.split("'")
                if len(parts) >= 3:
                    lbl = parts[1]
            elif '"' in user_text:
                parts = user_text.split('"')
                if len(parts) >= 3:
                    lbl = parts[1]

            return {
                "target_detected": True,
                "label": lbl,
                "bounding_box": coords,
                "frontier_score": None,
            }, True

        # Fallback default: non-detected
        return {
            "target_detected": False,
            "label": None,
            "bounding_box": None,
            "frontier_score": 0.85,
        }, False

    @staticmethod
    def export(
        dataset_manager,
        output_dir: str,
        format_type: str = "qwen",  # "qwen" (frontier JSON), "qwen_legacy", or "sharegpt"
        train_ratio: float = 0.8,
        split_method: str = "image",  # "image" (grouped) or "sequential"
        copy_images: bool = True,
        seed: int = 42,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Export dataset to JSONL files with strict data leakage prevention.

        split_method:
            - 'image': Splits by unique images (grouped). All prompts of an image stay in the same split.
            - 'sequential': Splits by sequential frame order (first N% train, remaining val).
                           Prevents temporal leakage from consecutive video frames.

        Returns:
            Tuple of (success: bool, message: str, stats: dict)
        """
        if not dataset_manager.project_dir:
            return False, "Project belum dipilih.", {}

        images = dataset_manager.get_image_list()
        if not images:
            return False, "Tidak ada gambar dalam project ini.", {}

        os.makedirs(output_dir, exist_ok=True)
        export_img_dir = os.path.join(output_dir, "images")
        if copy_images:
            os.makedirs(export_img_dir, exist_ok=True)

        # Group samples by image to prevent prompt-level leakage
        samples_by_image: Dict[str, List[Dict[str, Any]]] = {}
        positive_count = 0
        negative_count = 0

        for img_name in images:
            anno = dataset_manager.get_annotation(img_name)
            prompts = anno.get("prompts", [])
            boxes = anno.get("boxes", [])

            # If no manual prompts exist, create default prompt entries if boxes exist
            if not prompts and boxes:
                for b in boxes:
                    lbl = b.get("label", "object")
                    b_2d = b.get("box_2d", [0, 0, 0, 0])
                    payload = {
                        "target_detected": True,
                        "label": lbl,
                        "bounding_box": b_2d,
                        "frontier_score": None,
                    }
                    prompts.append({
                        "id": f"auto_{len(prompts)}",
                        "user": f"Cari objek '{lbl}' pada citra ini.",
                        "assistant": json.dumps(payload, ensure_ascii=False),
                    })

            if not prompts:
                continue

            # Determine image reference path in export
            if copy_images:
                src_path = dataset_manager.get_image_path(img_name)
                dst_path = os.path.join(export_img_dir, img_name)
                if os.path.isfile(src_path) and not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                image_ref = f"images/{img_name}"
            else:
                image_ref = dataset_manager.get_image_path(img_name)

            img_samples = []
            for idx, p in enumerate(prompts):
                u_text = p.get("user", "").strip()
                a_text = p.get("assistant", "").strip()
                if not u_text:
                    continue

                payload_dict, is_positive = DatasetExporter._normalize_frontier_payload(
                    assistant_text=a_text,
                    user_text=u_text,
                    boxes=boxes
                )

                if is_positive:
                    positive_count += 1
                else:
                    negative_count += 1

                sample_id = f"{os.path.splitext(img_name)[0]}_v{idx+1:02d}"
                assistant_json_str = json.dumps(payload_dict, ensure_ascii=False)

                if format_type in ("qwen", "qwen_frontier"):
                    # Standard Qwen2-VL Multimodal JSON with Semantic Grounding & Frontier Score
                    sample = {
                        "id": sample_id,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image_ref},
                                    {"type": "text", "text": u_text},
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": [
                                    {"type": "text", "text": assistant_json_str},
                                ],
                            },
                        ],
                    }
                elif format_type == "qwen_legacy":
                    # Legacy flat string format
                    sample = {
                        "id": sample_id,
                        "image": image_ref,
                        "messages": [
                            {"role": "user", "content": f"<image>\n{u_text}"},
                            {"role": "assistant", "content": a_text},
                        ],
                    }
                else:  # sharegpt / llava format
                    sample = {
                        "id": sample_id,
                        "image": image_ref,
                        "conversations": [
                            {"from": "human", "value": f"<image>\n{u_text}"},
                            {"from": "gpt", "value": assistant_json_str},
                        ],
                    }
                img_samples.append(sample)

            if img_samples:
                samples_by_image[img_name] = img_samples

        if not samples_by_image:
            return False, "Tidak ada data prompt yang valid untuk diekspor. Silakan buat prompt di Menu 3.", {}

        # -------------------------------------------------------------
        # Data Leakage Prevention: Split by Image or Sequential Frames
        # -------------------------------------------------------------
        valid_images = list(samples_by_image.keys())

        if split_method == "sequential":
            # Natural sort/order: first N% images -> train, rest -> val
            split_idx = max(1, int(len(valid_images) * train_ratio))
            if split_idx >= len(valid_images) and len(valid_images) > 1:
                split_idx = len(valid_images) - 1
            train_images = set(valid_images[:split_idx])
            val_images = set(valid_images[split_idx:])
        else:
            # Grouped by image with random shuffle of images (NOT samples)
            rng = random.Random(seed)
            shuffled_images = list(valid_images)
            rng.shuffle(shuffled_images)

            split_idx = max(1, int(len(shuffled_images) * train_ratio))
            if split_idx >= len(shuffled_images) and len(shuffled_images) > 1:
                split_idx = len(shuffled_images) - 1
            train_images = set(shuffled_images[:split_idx])
            val_images = set(shuffled_images[split_idx:])

        train_samples: List[Dict[str, Any]] = []
        val_samples: List[Dict[str, Any]] = []

        for img_name in valid_images:
            if img_name in train_images:
                train_samples.extend(samples_by_image[img_name])
            else:
                val_samples.extend(samples_by_image[img_name])

        # Write out train.jsonl and val.jsonl
        train_file = os.path.join(output_dir, "train.jsonl")
        val_file = os.path.join(output_dir, "val.jsonl")

        with open(train_file, "w", encoding="utf-8") as f:
            for s in train_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        with open(val_file, "w", encoding="utf-8") as f:
            for s in val_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        total_samples = len(train_samples) + len(val_samples)

        # Save metadata summary
        stats = {
            "total_images": len(valid_images),
            "train_images": len(train_images),
            "val_images": len(val_images),
            "total_samples": total_samples,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "positive_samples": positive_count,
            "negative_samples_null": negative_count,
            "format": format_type,
            "split_method": split_method,
            "train_ratio": train_ratio,
        }

        info_file = os.path.join(output_dir, "dataset_info.json")
        with open(info_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        msg = (
            f"Berhasil mengekspor {total_samples} sampel dari {len(valid_images)} gambar.\n"
            f"• Train: {len(train_samples)} sampel ({len(train_images)} gambar)\n"
            f"• Val: {len(val_samples)} sampel ({len(val_images)} gambar)\n"
            f"• Metode Split: {split_method.capitalize()} (Bebas kebocoran data)"
        )
        return True, msg, stats
