import os
import unittest
import json
from core.prompt_generator import TemplatePromptGenerator

class TestTemplatePromptGenerator(unittest.TestCase):

    def setUp(self):
        self.gen = TemplatePromptGenerator()

    def test_templates_loaded(self):
        id_tpls = self.gen.get_templates("id")
        en_tpls = self.gen.get_templates("en")
        both_tpls = self.gen.get_templates("both")

        self.assertEqual(len(id_tpls), 50)
        self.assertEqual(len(en_tpls), 50)
        self.assertEqual(len(both_tpls), 100)

        for t in both_tpls:
            self.assertIn("X", t)

    def test_positive_and_negative_prompts(self):
        project_classes = ["kursi", "kardus", "dispenser", "door", "apar", "human"]
        boxes = [
            {"label": "kursi", "box_2d": [100, 150, 400, 450]},
            {"label": "dispenser", "box_2d": [250, 120, 680, 450]},
        ]

        # Generate with max 2 templates per class
        prompts = self.gen.generate_frame_prompts(
            boxes=boxes,
            project_classes=project_classes,
            language="id",
            max_templates_per_class=2,
            frontier_score=0.85,
        )

        # 2 detected classes (kursi, dispenser) * 2 = 4 positive prompts
        # 4 missing classes (kardus, door, apar, human) * 2 = 8 negative prompts
        # Total = 12 prompts
        self.assertEqual(len(prompts), 12)

        pos_count = 0
        neg_count = 0
        for p in prompts:
            data = json.loads(p["assistant"])
            if data["target_detected"]:
                pos_count += 1
                self.assertIn(data["label"], ["kursi", "dispenser"])
                self.assertIsNotNone(data["bounding_box"])
                self.assertIsNotNone(data["frontier_score"])
                self.assertTrue(0.15 <= data["frontier_score"] <= 0.95)
            else:
                neg_count += 1
                self.assertIsNone(data["label"])
                self.assertIsNone(data["bounding_box"])
                self.assertEqual(data["frontier_score"], 0.85)

        self.assertEqual(pos_count, 4)
        self.assertEqual(neg_count, 8)

    def test_multiple_instances_of_same_class(self):
        project_classes = ["kardus", "door"]
        boxes = [
            {"label": "kardus", "box_2d": [10, 10, 50, 50]},
            {"label": "kardus", "box_2d": [60, 60, 90, 90]},
        ]

        prompts = self.gen.generate_frame_prompts(
            boxes=boxes,
            project_classes=project_classes,
            language="id",
            max_templates_per_class=2,
            frontier_score=0.75,
        )

        # 2 boxes of kardus * 2 = 4 positive prompts
        # 1 missing class (door) * 2 = 2 negative prompts
        self.assertEqual(len(prompts), 6)

        # Check that both boxes are represented in positive prompts
        target_boxes = [json.loads(p["assistant"])["bounding_box"] for p in prompts if json.loads(p["assistant"])["target_detected"]]
        self.assertIn([10, 10, 50, 50], target_boxes)
        self.assertIn([60, 60, 90, 90], target_boxes)

    def test_both_languages_and_duplicates(self):
        project_classes = ["dispenser"]
        boxes = [{"label": "dispenser", "box_2d": [100, 200, 300, 400]}]

        # Pre-existing prompt
        existing = [{
            "id": "p_1",
            "user": "Tolong cari dispenser di ruangan ini.",
            "assistant": json.dumps({"target_detected": True, "label": "dispenser", "bounding_box": [100, 200, 300, 400], "frontier_score": None})
        }]

        prompts = self.gen.generate_frame_prompts(
            boxes=boxes,
            project_classes=project_classes,
            language="both",
            existing_prompts=existing,
        )

        # 100 total templates (50 ID + 50 EN) minus 1 duplicate = 99 new prompts
        self.assertEqual(len(prompts), 99)
        users = [p["user"] for p in prompts]
        self.assertNotIn("Tolong cari dispenser di ruangan ini.", users)
        self.assertIn("Find dispenser in this room.", users)

    def test_empty_boxes_all_negative(self):
        project_classes = ["kursi", "meja"]
        boxes = []

        prompts = self.gen.generate_frame_prompts(
            boxes=boxes,
            project_classes=project_classes,
            language="id",
            max_templates_per_class=3,
            frontier_score=0.9,
        )

        self.assertEqual(len(prompts), 6)
        for p in prompts:
            data = json.loads(p["assistant"])
            self.assertFalse(data["target_detected"])
            self.assertEqual(data["frontier_score"], 0.9)

    def test_export_integration(self):
        import tempfile
        import shutil
        import os
        from core.dataset_manager import DatasetManager
        from core.exporter import DatasetExporter

        temp_dir = tempfile.mkdtemp()
        try:
            dm = DatasetManager(temp_dir)
            dm.classes = ["dispenser", "kursi", "door"]

            # Create dummy image
            img_name = "frame_001.jpg"
            img_path = os.path.join(dm.frames_dir, img_name)
            with open(img_path, "wb") as f:
                f.write(b"dummy")

            # 1 box of dispenser
            anno = dm.get_annotation(img_name)
            anno["boxes"] = [{"label": "dispenser", "box_2d": [100, 200, 300, 400]}]

            # Generate prompts using template generator
            anno["prompts"] = self.gen.generate_frame_prompts(
                boxes=anno["boxes"],
                project_classes=dm.classes,
                language="id",
                max_templates_per_class=2,
                frontier_score=0.85,
            )
            dm.save_annotation(img_name, anno)

            export_dir = os.path.join(temp_dir, "export_test")
            ok, msg, stats = DatasetExporter.export(
                dataset_manager=dm,
                output_dir=export_dir,
                format_type="qwen",
                train_ratio=1.0,
                copy_images=False,
            )

            self.assertTrue(ok)
            # 1 positive class * 2 = 2 positive samples
            # 2 negative classes (kursi, door) * 2 = 4 negative samples
            self.assertEqual(stats["total_samples"], 6)
            self.assertEqual(stats["positive_samples"], 2)
            self.assertEqual(stats["negative_samples_null"], 4)

            # Verify exported line format
            train_file = os.path.join(export_dir, "train.jsonl")
            self.assertTrue(os.path.isfile(train_file))
            with open(train_file, "r") as f:
                lines = [json.loads(l) for l in f]
                self.assertEqual(len(lines), 6)
                for item in lines:
                    msg = item["messages"]
                    self.assertEqual(msg[0]["role"], "user")
                    self.assertEqual(msg[1]["role"], "assistant")
                    ast_content = msg[1]["content"][0]["text"]
                    parsed = json.loads(ast_content)
                    self.assertIn("target_detected", parsed)
        finally:
            shutil.rmtree(temp_dir)

    def test_clear_all_dataset_prompts_logic(self):
        import tempfile
        import shutil
        from core.dataset_manager import DatasetManager

        temp_dir = tempfile.mkdtemp()
        try:
            dm = DatasetManager(temp_dir)
            for i in range(3):
                name = f"img_{i}.jpg"
                with open(os.path.join(dm.frames_dir, name), "wb") as f:
                    f.write(b"data")
                anno = dm.get_annotation(name)
                anno["prompts"] = [{"id": "p1", "user": "test", "assistant": "{}"}]
                dm.save_annotation(name, anno)

            # Verify initially all have prompts
            for name in dm.get_image_list():
                self.assertEqual(len(dm.get_annotation(name).get("prompts", [])), 1)

            # Clear all
            for name in dm.get_image_list():
                anno = dm.get_annotation(name)
                anno["prompts"] = []
                dm.save_annotation(name, anno)

            # Verify all cleared
            for name in dm.get_image_list():
                self.assertEqual(len(dm.get_annotation(name).get("prompts", [])), 0)
        finally:
            shutil.rmtree(temp_dir)

    def test_synonym_substitution_in_prompts(self):
        project_classes = ["trash_bin", "fire_extinguisher"]
        boxes = [
            {"label": "trash_bin", "box_2d": [100, 200, 300, 400]}
        ]
        class_synonyms = {
            "trash_bin": {
                "id": ["tempat sampah", "tong sampah"],
                "en": ["trash can", "garbage bin"],
            },
            "fire_extinguisher": {
                "id": ["alat pemadam api", "tabung apar"],
                "en": ["fire extinguisher cylinder"],
            }
        }

        # 1. Test Indonesian prompts
        prompts_id = self.gen.generate_frame_prompts(
            boxes=boxes,
            project_classes=project_classes,
            language="id",
            max_templates_per_class=4,
            class_synonyms=class_synonyms,
        )
        self.assertTrue(len(prompts_id) > 0)
        for p in prompts_id:
            ast = json.loads(p["assistant"])
            user_text = p["user"]
            if ast["target_detected"]:
                # Positive prompt: Assistant label must be canonical
                self.assertEqual(ast["label"], "trash_bin")
                # User prompt text must contain one of ID synonyms
                has_synonym = any(s in user_text.lower() for s in ["tempat sampah", "tong sampah"])
                self.assertTrue(has_synonym, f"Expected ID synonym in prompt, got: {user_text}")
                # Must not contain English synonyms
                self.assertNotIn("trash can", user_text.lower())
                self.assertNotIn("garbage bin", user_text.lower())
            else:
                # Negative prompt (fire_extinguisher)
                self.assertIsNone(ast["label"])
                has_synonym = any(s in user_text.lower() for s in ["alat pemadam api", "tabung apar"])
                self.assertTrue(has_synonym, f"Expected negative ID synonym in prompt, got: {user_text}")

        # 2. Test English prompts
        prompts_en = self.gen.generate_frame_prompts(
            boxes=boxes,
            project_classes=project_classes,
            language="en",
            max_templates_per_class=4,
            class_synonyms=class_synonyms,
        )
        self.assertTrue(len(prompts_en) > 0)
        for p in prompts_en:
            ast = json.loads(p["assistant"])
            user_text = p["user"]
            if ast["target_detected"]:
                self.assertEqual(ast["label"], "trash_bin")
                has_synonym = any(s in user_text.lower() for s in ["trash can", "garbage bin"])
                self.assertTrue(has_synonym, f"Expected EN synonym in prompt, got: {user_text}")
                self.assertNotIn("tempat sampah", user_text.lower())
            else:
                self.assertIsNone(ast["label"])
                self.assertIn("fire extinguisher cylinder", user_text.lower())

    def test_dynamic_frontier_score_calculation(self):
        # 1. Tiny box far away -> score near 0.95
        tiny_box = [100, 100, 140, 140]  # 40x40
        score_tiny = TemplatePromptGenerator.calculate_dynamic_frontier_score(tiny_box)
        self.assertGreaterEqual(score_tiny, 0.85)
        self.assertLessEqual(score_tiny, 0.95)

        # 2. Medium box -> score around 0.50 - 0.70
        med_box = [200, 200, 450, 450]  # 250x250
        score_med = TemplatePromptGenerator.calculate_dynamic_frontier_score(med_box)
        self.assertGreater(score_tiny, score_med)
        self.assertTrue(0.50 <= score_med <= 0.70)

        # 3. Huge box close up -> score drops to minimum 0.15
        huge_box = [100, 100, 750, 750]  # 650x650
        score_huge = TemplatePromptGenerator.calculate_dynamic_frontier_score(huge_box)
        self.assertEqual(score_huge, 0.15)

        # 4. Monotonicity: larger box must have lower or equal frontier score
        self.assertGreater(score_tiny, score_med)
        self.assertGreater(score_med, score_huge)

    def test_frontier_modes_dynamic_vs_static(self):
        project_classes = ["pallet"]
        boxes = [{"label": "pallet", "box_2d": [100, 100, 200, 200]}]

        # Dynamic mode
        dynamic_prompts = self.gen.generate_frame_prompts(
            boxes=boxes,
            project_classes=project_classes,
            language="id",
            frontier_mode="dynamic"
        )
        for p in dynamic_prompts:
            ast = json.loads(p["assistant"])
            if ast["target_detected"]:
                expected = TemplatePromptGenerator.calculate_dynamic_frontier_score([100, 100, 200, 200])
                self.assertEqual(ast["frontier_score"], expected)

        # Static mode
        static_prompts = self.gen.generate_frame_prompts(
            boxes=boxes,
            project_classes=project_classes,
            language="id",
            frontier_mode="static",
            positive_frontier_score=0.72
        )
        for p in static_prompts:
            ast = json.loads(p["assistant"])
            if ast["target_detected"]:
                self.assertEqual(ast["frontier_score"], 0.72)

        # Null mode
        null_prompts = self.gen.generate_frame_prompts(
            boxes=boxes,
            project_classes=project_classes,
            language="id",
            frontier_mode="null"
        )
        for p in null_prompts:
            ast = json.loads(p["assistant"])
            if ast["target_detected"]:
                self.assertIsNone(ast["frontier_score"])
            else:
                self.assertIsNotNone(ast["frontier_score"])

if __name__ == "__main__":
    unittest.main()
