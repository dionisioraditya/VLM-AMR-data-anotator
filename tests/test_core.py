import os
import json
import shutil
import tempfile
import unittest
from core.dataset_manager import DatasetManager
from core.exporter import DatasetExporter
from core.gemini_client import GeminiClient
from core.video_extractor import VideoExtractor
from core.prompt_generator import TemplatePromptGenerator
import cv2
import numpy as np

class TestCoreModules(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dm = DatasetManager(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_coordinate_transforms(self):
        w, h = 1920, 1080
        # Given pixel box [x=100, y=200, w=300, h=400]
        pixel_box = [100, 200, 300, 400]
        norm_box = DatasetManager.normalize_box(pixel_box, w, h)
        self.assertEqual(len(norm_box), 4)

        # ymin, xmin, ymax, xmax
        ymin, xmin, ymax, xmax = norm_box
        self.assertTrue(0 <= ymin <= ymax <= 1000)
        self.assertTrue(0 <= xmin <= xmax <= 1000)

        # Denormalize back
        denorm = DatasetManager.denormalize_box(norm_box, w, h)
        # Allow slight rounding difference within 2 pixels
        self.assertAlmostEqual(denorm[0], 100, delta=2)
        self.assertAlmostEqual(denorm[1], 200, delta=2)
        self.assertAlmostEqual(denorm[2], 300, delta=2)
        self.assertAlmostEqual(denorm[3], 400, delta=2)

    def test_dataset_manager_annotation(self):
        img_name = "test_frame_01.jpg"
        img_path = os.path.join(self.dm.frames_dir, img_name)
        with open(img_path, "wb") as f:
            f.write(b"dummy image data")

        # Initial annotation
        anno = self.dm.get_annotation(img_name)
        self.assertEqual(anno["image_file"], img_name)
        self.assertEqual(len(anno["boxes"]), 0)

        # Save box
        anno["boxes"].append({
            "id": "box_1",
            "label": "pallet",
            "box_2d": [600, 300, 800, 500]
        })
        self.dm.save_annotation(img_name, anno)

        self.assertTrue(self.dm.is_annotated(img_name))

        # Reload
        loaded = self.dm.get_annotation(img_name)
        self.assertEqual(len(loaded["boxes"]), 1)
        self.assertEqual(loaded["boxes"][0]["label"], "pallet")
        self.assertEqual(loaded["boxes"][0]["box_2d"], [600, 300, 800, 500])

    def test_rename_class_and_sync_annotations(self):
        # 1. Setup classes
        self.dm.add_class("trash_bin")
        self.assertIn("trash_bin", self.dm.classes)

        # 2. Create annotations using 'trash_bin'
        img1 = "img_01.jpg"
        img2 = "img_02.jpg"
        for name in [img1, img2]:
            with open(os.path.join(self.dm.frames_dir, name), "wb") as f:
                f.write(b"dummy")
            anno = self.dm.get_annotation(name)
            anno["boxes"] = [
                {"id": "b1", "label": "trash_bin", "box_2d": [100, 100, 200, 200]},
                {"id": "b2", "label": "door", "box_2d": [300, 300, 400, 400]}
            ]
            self.dm.save_annotation(name, anno)

        # 3. Rename class 'trash_bin' -> 'tempat_sampah'
        updated_count = self.dm.rename_class("trash_bin", "tempat_sampah", update_annotations=True)
        self.assertEqual(updated_count, 2)

        # 4. Verify class list updated
        self.assertNotIn("trash_bin", self.dm.classes)
        self.assertIn("tempat_sampah", self.dm.classes)

        # 5. Verify annotations updated
        anno1 = self.dm.get_annotation(img1)
        anno2 = self.dm.get_annotation(img2)
        self.assertEqual(anno1["boxes"][0]["label"], "tempat_sampah")
        self.assertEqual(anno1["boxes"][1]["label"], "door")
        self.assertEqual(anno2["boxes"][0]["label"], "tempat_sampah")

    def test_class_synonyms_management(self):
        # Default synonyms should be populated
        syns_trash = self.dm.get_class_synonyms("trash_bin", "id")
        self.assertIn("tempat sampah", syns_trash)
        self.assertIn("tong sampah", syns_trash)

        syns_trash_en = self.dm.get_class_synonyms("trash_bin", "en")
        self.assertIn("trash can", syns_trash_en)

        # Set custom synonyms
        self.dm.set_class_synonyms("trash_bin", "id", ["bak sampah", "tong sampah"])
        self.assertEqual(self.dm.get_class_synonyms("trash_bin", "id"), ["bak sampah", "tong sampah"])

        # Rename class should carry over synonyms
        self.dm.rename_class("trash_bin", "wadah_sampah", update_annotations=False)
        self.assertEqual(self.dm.get_class_synonyms("wadah_sampah", "id"), ["bak sampah", "tong sampah"])
        self.assertIn("trash can", self.dm.get_class_synonyms("wadah_sampah", "en"))
        self.assertEqual(self.dm.get_class_synonyms("trash_bin", "id"), [])

        # Persistence in config
        dm_reloaded = DatasetManager(self.test_dir)
        self.assertEqual(dm_reloaded.get_class_synonyms("wadah_sampah", "id"), ["bak sampah", "tong sampah"])

    def test_exporter_qwen_and_sharegpt(self):
        img_name = "frame_001.jpg"
        img_path = os.path.join(self.dm.frames_dir, img_name)
        with open(img_path, "wb") as f:
            f.write(b"fake image bytes")

        anno = self.dm.get_annotation(img_name)
        anno["boxes"] = [
            {"label": "pallet", "box_2d": [620, 310, 850, 540]}
        ]
        anno["prompts"] = [
            {"id": "p1", "user": "Deteksi pallet", "assistant": "[620, 310, 850, 540]"},
            {"id": "p2", "user": "Deteksi forklift", "assistant": "null"}  # Negative sample
        ]
        self.dm.save_annotation(img_name, anno)

        # Test Qwen Export
        qwen_out = os.path.join(self.test_dir, "export_qwen")
        ok, msg, stats = DatasetExporter.export(
            dataset_manager=self.dm,
            output_dir=qwen_out,
            format_type="qwen",
            train_ratio=1.0,
            copy_images=False,
        )
        self.assertTrue(ok)
        self.assertEqual(stats["total_samples"], 2)
        self.assertEqual(stats["positive_samples"], 1)
        self.assertEqual(stats["negative_samples_null"], 1)

        # Read train.jsonl
        train_file = os.path.join(qwen_out, "train.jsonl")
        self.assertTrue(os.path.isfile(train_file))
        with open(train_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        self.assertEqual(len(lines), 2)

        # Verify Qwen2-VL Multimodal Content Structure & Frontier Payload
        has_positive = False
        has_negative = False
        for l in lines:
            entry = json.loads(l)
            self.assertIn("id", entry)
            self.assertIn("messages", entry)

            user_content = entry["messages"][0]["content"]
            self.assertEqual(user_content[0]["type"], "image")
            self.assertEqual(user_content[1]["type"], "text")

            assistant_content = entry["messages"][1]["content"]
            self.assertEqual(assistant_content[0]["type"], "text")

            payload = json.loads(assistant_content[0]["text"])
            self.assertIn("target_detected", payload)
            if payload["target_detected"]:
                has_positive = True
                self.assertIsNotNone(payload["frontier_score"])
                self.assertTrue(0.15 <= payload["frontier_score"] <= 0.95)
            else:
                has_negative = True
                self.assertIsNone(payload["bounding_box"])
                self.assertIsNotNone(payload["frontier_score"])
                self.assertAlmostEqual(payload["frontier_score"], 0.85)

        self.assertTrue(has_positive)
        self.assertTrue(has_negative)

    def test_exporter_paligemma(self):
        # 1. Test coordinate tokenization
        self.assertEqual(DatasetExporter.to_paligemma_loc(0), "<loc0000>")
        self.assertEqual(DatasetExporter.to_paligemma_loc(1000), "<loc1023>")
        self.assertEqual(DatasetExporter.to_paligemma_loc(500), "<loc0512>")

        # 2. Test dataset export
        img_name = "frame_paligemma_01.jpg"
        img_path = os.path.join(self.dm.frames_dir, img_name)
        with open(img_path, "wb") as f:
            f.write(b"dummy paligemma frame")

        anno = self.dm.get_annotation(img_name)
        anno["boxes"] = [
            {"label": "dispenser", "box_2d": [250, 120, 680, 450]}
        ]
        anno["prompts"] = [
            {
                "id": "p_pos",
                "user": "Cari objek 'dispenser' pada citra ini.",
                "assistant": json.dumps({
                    "target_detected": True,
                    "label": "dispenser",
                    "bounding_box": [250, 120, 680, 450],
                    "frontier_score": None
                })
            },
            {
                "id": "p_neg",
                "user": "Cari objek 'kursi' pada citra ini.",
                "assistant": json.dumps({
                    "target_detected": False,
                    "label": None,
                    "bounding_box": None,
                    "frontier_score": 0.85
                })
            }
        ]
        self.dm.save_annotation(img_name, anno)

        pali_out = os.path.join(self.test_dir, "export_paligemma")
        ok, msg, stats = DatasetExporter.export(
            dataset_manager=self.dm,
            output_dir=pali_out,
            format_type="paligemma",
            train_ratio=1.0,
            copy_images=False,
        )
        self.assertTrue(ok)
        self.assertEqual(stats["total_samples"], 2)
        self.assertEqual(stats["format"], "paligemma")

        train_file = os.path.join(pali_out, "train.jsonl")
        self.assertTrue(os.path.isfile(train_file))
        with open(train_file, "r") as f:
            lines = [json.loads(l.strip()) for l in f if l.strip()]
        self.assertEqual(len(lines), 2)

        # Check positive entry
        pos_entry = next(e for e in lines if "dispenser" in e["suffix"])
        self.assertIn("id", pos_entry)
        self.assertIn("image", pos_entry)
        self.assertEqual(pos_entry["prefix"], "Cari objek 'dispenser' pada citra ini.")
        # [250, 120, 680, 450] -> 250/1000*1023=256, 120/1000*1023=123, 680/1000*1023=696, 450/1000*1023=460
        self.assertEqual(pos_entry["suffix"], "<loc0256><loc0123><loc0696><loc0460> dispenser")

        # Check negative entry with frontier score
        neg_entry = next(e for e in lines if "frontier" in e["suffix"])
        self.assertEqual(neg_entry["prefix"], "Cari objek 'kursi' pada citra ini.")
        self.assertEqual(neg_entry["suffix"], "none ; frontier: 0.85")

    def test_data_leakage_prevention(self):
        # Create 4 images with multiple prompt variations each
        for i in range(1, 5):
            img_name = f"frame_{i:03d}.jpg"
            img_path = os.path.join(self.dm.frames_dir, img_name)
            with open(img_path, "wb") as f:
                f.write(b"fake image")

            anno = self.dm.get_annotation(img_name)
            anno["boxes"] = [{"label": "dispenser", "box_2d": [100, 200, 300, 400]}]
            anno["prompts"] = [
                {"id": f"p_{i}_1", "user": f"Prompt 1 for img {i}", "assistant": "[100, 200, 300, 400]"},
                {"id": f"p_{i}_2", "user": f"Prompt 2 for img {i}", "assistant": "[100, 200, 300, 400]"},
            ]
            self.dm.save_annotation(img_name, anno)

        out_dir = os.path.join(self.test_dir, "export_leakage_test")
        ok, msg, stats = DatasetExporter.export(
            dataset_manager=self.dm,
            output_dir=out_dir,
            train_ratio=0.5,
            split_method="image",
            copy_images=False,
        )
        self.assertTrue(ok)

        # Extract images in train.jsonl and val.jsonl
        train_imgs = set()
        with open(os.path.join(out_dir, "train.jsonl"), "r") as f:
            for line in f:
                data = json.loads(line)
                if "messages" in data and isinstance(data["messages"][0]["content"], list):
                    train_imgs.add(data["messages"][0]["content"][0]["image"])
                else:
                    train_imgs.add(data["image"])

        val_imgs = set()
        with open(os.path.join(out_dir, "val.jsonl"), "r") as f:
            for line in f:
                data = json.loads(line)
                if "messages" in data and isinstance(data["messages"][0]["content"], list):
                    val_imgs.add(data["messages"][0]["content"][0]["image"])
                else:
                    val_imgs.add(data["image"])

        # Strictly verify no intersection between train and val images (NO LEAKAGE)
        intersection = train_imgs.intersection(val_imgs)
        self.assertEqual(len(intersection), 0, f"Data leakage detected! Images in both train and val: {intersection}")
        self.assertEqual(len(train_imgs) + len(val_imgs), 4)

    def test_gemini_client_backends(self):
        # Test backend switching and default models
        client = GeminiClient(backend=GeminiClient.BACKEND_AGY)
        self.assertEqual(client.backend, GeminiClient.BACKEND_AGY)
        self.assertIn(client.model, GeminiClient.AGY_MODELS)

        client.set_backend(GeminiClient.BACKEND_API)
        self.assertEqual(client.backend, GeminiClient.BACKEND_API)
        self.assertIn(client.model, GeminiClient.API_MODELS)

        client.set_backend(GeminiClient.BACKEND_AGY)
        self.assertEqual(client.backend, GeminiClient.BACKEND_AGY)
        self.assertIn(client.model, GeminiClient.AGY_MODELS)

        agy_path = GeminiClient.find_agy_path()
        self.assertTrue(agy_path is None or os.path.exists(agy_path))

    def test_batch_worker_unannotated_filter(self):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PySide6.QtCore import QCoreApplication
        app = QCoreApplication.instance() or QCoreApplication([])

        from ui.tab_annotate import GeminiBatchWorker

        # Create 3 images
        img_names = ["frame_01.jpg", "frame_02.jpg", "frame_03.jpg"]
        for img in img_names:
            p = os.path.join(self.dm.frames_dir, img)
            with open(p, "wb") as f:
                f.write(b"dummy")

        # Pre-annotate frame_01 and frame_03
        anno1 = self.dm.get_annotation("frame_01.jpg")
        anno1["boxes"] = [{"label": "box_already", "box_2d": [10, 10, 50, 50]}]
        self.dm.save_annotation("frame_01.jpg", anno1)

        anno3 = self.dm.get_annotation("frame_03.jpg")
        anno3["boxes"] = [{"label": "box_already", "box_2d": [20, 20, 60, 60]}]
        self.dm.save_annotation("frame_03.jpg", anno3)

        self.assertTrue(self.dm.is_annotated("frame_01.jpg"))
        self.assertFalse(self.dm.is_annotated("frame_02.jpg"))
        self.assertTrue(self.dm.is_annotated("frame_03.jpg"))

        # Mock gemini client
        detected_calls = []

        class MockGeminiClient:
            backend = GeminiClient.BACKEND_AGY
            model = "gemini-3.8-flash-high"

            def detect_objects(self, image_path, target_classes=None, custom_instructions=None):
                detected_calls.append(os.path.basename(image_path))
                return True, "Mock detected", [{"label": "pallet", "box_2d": [100, 100, 400, 400]}]

        mock_client = MockGeminiClient()
        worker = GeminiBatchWorker(
            gemini_client=mock_client,
            dataset_manager=self.dm,
            image_names=img_names,
            only_unannotated=True,
            target_classes=["pallet"],
            delay_seconds=0.0,
        )

        progress_msgs = []
        worker.progress.connect(lambda cur, tot, img, status: progress_msgs.append((img, status)))
        worker.run()

        # Only frame_02.jpg should have been passed to detect_objects
        self.assertEqual(detected_calls, ["frame_02.jpg"])

        # frame_01 and frame_03 should have logged "Melewati"
        skip_msgs = [status for img, status in progress_msgs if "Melewati" in status]
        self.assertEqual(len(skip_msgs), 2)

        # Now all 3 are annotated
        self.assertTrue(self.dm.is_annotated("frame_01.jpg"))
        self.assertTrue(self.dm.is_annotated("frame_02.jpg"))
        self.assertTrue(self.dm.is_annotated("frame_03.jpg"))

    def test_video_extractor_batch(self):
        # 1. Create two small synthetic test videos
        v1_path = os.path.join(self.test_dir, "test_vid_1.mp4")
        v2_path = os.path.join(self.test_dir, "test_vid_2.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        for v_path, num_frames in [(v1_path, 10), (v2_path, 10)]:
            writer = cv2.VideoWriter(v_path, fourcc, 10, (64, 64))
            for i in range(num_frames):
                img = np.zeros((64, 64, 3), dtype=np.uint8)
                img[:] = (i * 20) % 255
                writer.write(img)
            writer.release()
            self.assertTrue(os.path.isfile(v_path))

        extractor = VideoExtractor()

        # 2. Test batch extraction with subfolders
        batch_out_sub = os.path.join(self.test_dir, "batch_out_sub")
        progress_records = []

        def on_batch_prog(cur_v, tot_v, vid_name, cur_f, tot_f, saved):
            progress_records.append((cur_v, tot_v, vid_name, saved))

        ok, msg, total_saved = extractor.extract_batch(
            video_paths=[v1_path, v2_path],
            output_dir=batch_out_sub,
            create_subfolders=True,
            mode="frame",
            frame_interval=2,
            prefix="amr",
            progress_callback=on_batch_prog,
        )

        self.assertTrue(ok)
        self.assertGreater(total_saved, 0)
        self.assertTrue(len(progress_records) > 0)

        # Check subfolders
        sub1 = os.path.join(batch_out_sub, "test_vid_1")
        sub2 = os.path.join(batch_out_sub, "test_vid_2")
        self.assertTrue(os.path.isdir(sub1))
        self.assertTrue(os.path.isdir(sub2))
        files1 = os.listdir(sub1)
        files2 = os.listdir(sub2)
        self.assertEqual(len(files1), 5)  # 10 frames / interval 2 = 5
        self.assertEqual(len(files2), 5)
        self.assertTrue(all(f.startswith("amr_") for f in files1))

        # 3. Test batch extraction without subfolders (shared directory)
        batch_out_flat = os.path.join(self.test_dir, "batch_out_flat")
        ok_flat, msg_flat, total_saved_flat = extractor.extract_batch(
            video_paths=[v1_path, v2_path],
            output_dir=batch_out_flat,
            create_subfolders=False,
            mode="frame",
            frame_interval=2,
            prefix="amr",
        )
        self.assertTrue(ok_flat)
        self.assertEqual(total_saved_flat, 10)
        flat_files = os.listdir(batch_out_flat)
        self.assertEqual(len(flat_files), 10)
        # Verify names are differentiated by video name
        self.assertTrue(any("test_vid_1" in f for f in flat_files))
        self.assertTrue(any("test_vid_2" in f for f in flat_files))

        # 4. Test cancel during batch
        def cancel_cb(*args):
            extractor.cancel()

        ok_c, msg_c, saved_c = extractor.extract_batch(
            video_paths=[v1_path, v2_path],
            output_dir=os.path.join(self.test_dir, "batch_cancelled"),
            create_subfolders=True,
            progress_callback=cancel_cb,
        )
        self.assertFalse(ok_c)
        self.assertIn("dibatalkan", msg_c)

    def test_auto_split_proportions_and_stratification(self):
        # 1. Create 10 dummy images with annotations
        # 6 pallet, 4 forklift
        for i in range(1, 11):
            img_name = f"split_test_{i:02d}.jpg"
            with open(os.path.join(self.dm.frames_dir, img_name), "wb") as f:
                f.write(b"data")
            anno = self.dm.get_annotation(img_name)
            lbl = "pallet" if i <= 6 else "forklift"
            anno["boxes"] = [{"label": lbl, "box_2d": [100, 100, 200, 200]}]
            self.dm.save_annotation(img_name, anno)

        # 2. Run auto_split: 70% Train (7), 20% Val (2), 10% Test (1)
        res = self.dm.auto_split(train_pct=70, val_pct=20, test_pct=10, stratify=True, random_seed=42)
        self.assertEqual(len(res["train"]), 7)
        self.assertEqual(len(res["val"]), 2)
        self.assertEqual(len(res["test"]), 1)

        # Verify no overlap
        train_s = set(res["train"])
        val_s = set(res["val"])
        test_s = set(res["test"])
        self.assertEqual(len(train_s.intersection(val_s)), 0)
        self.assertEqual(len(train_s.intersection(test_s)), 0)
        self.assertEqual(len(val_s.intersection(test_s)), 0)

        # 3. Test move_image_to_split
        target_img = res["train"][0]
        self.dm.move_image_to_split(target_img, "test")
        splits_after_move = self.dm.get_splits()
        self.assertNotIn(target_img, splits_after_move["train"])
        self.assertIn(target_img, splits_after_move["test"])
        self.assertEqual(self.dm.get_split_for_image(target_img), "test")

        # 4. Test persistence in project_config.json
        dm_reloaded = DatasetManager(self.test_dir)
        self.assertEqual(dm_reloaded.get_splits(), splits_after_move)

        # 5. Test delete_image cleans up from split
        self.dm.delete_image(target_img)
        splits_after_del = self.dm.get_splits()
        self.assertNotIn(target_img, splits_after_del["test"])

    def test_template_isolation_and_crud(self):
        # 1. Test master template paths for train, val, test
        train_path = TemplatePromptGenerator.get_default_template_path_for_split("train")
        val_path = TemplatePromptGenerator.get_default_template_path_for_split("val")
        test_path = TemplatePromptGenerator.get_default_template_path_for_split("test")

        self.assertTrue(os.path.isfile(train_path))
        self.assertTrue(os.path.isfile(val_path))
        self.assertTrue(os.path.isfile(test_path))
        self.assertNotEqual(train_path, val_path)
        self.assertNotEqual(train_path, test_path)
        self.assertNotEqual(val_path, test_path)

        # 2. Test isolation between generators
        gen_train = TemplatePromptGenerator(split="train")
        gen_val = TemplatePromptGenerator(split="val")
        gen_test = TemplatePromptGenerator(split="test")

        self.assertGreater(len(gen_train.id_templates), 0)
        self.assertGreater(len(gen_val.id_templates), 0)
        self.assertGreater(len(gen_test.id_templates), 0)

        # Train templates (100) should be distinct from val templates
        self.assertNotEqual(set(gen_train.id_templates), set(gen_val.id_templates))
        self.assertNotEqual(set(gen_train.id_templates), set(gen_test.id_templates))

        # 3. Test project-isolated template files in DatasetManager
        p_train_path = self.dm.get_template_path_for_split("train")
        p_val_path = self.dm.get_template_path_for_split("val")
        p_test_path = self.dm.get_template_path_for_split("test")
        self.assertTrue(os.path.isfile(p_train_path))
        self.assertTrue(os.path.isfile(p_val_path))
        self.assertTrue(os.path.isfile(p_test_path))

        # 4. Test CRUD operations on template generator
        custom_tpl_path = os.path.join(self.test_dir, "custom_templates.txt")
        gen_custom = TemplatePromptGenerator(split="train")
        # Add
        added = gen_custom.add_template("Tolong temukan X sekarang!", lang="id")
        self.assertTrue(added)
        self.assertIn("Tolong temukan X sekarang!", gen_custom.id_templates)

        # Edit
        edited = gen_custom.edit_template("Tolong temukan X sekarang!", "Segera temukan X!", lang="id")
        self.assertTrue(edited)
        self.assertIn("Segera temukan X!", gen_custom.id_templates)
        self.assertNotIn("Tolong temukan X sekarang!", gen_custom.id_templates)

        # Remove
        removed = gen_custom.remove_template("Segera temukan X!", lang="id")
        self.assertTrue(removed)
        self.assertNotIn("Segera temukan X!", gen_custom.id_templates)

        # Save and Reload
        gen_custom.add_template("Uji coba template X berhasil.", lang="id")
        gen_custom.save_templates_to_file(custom_tpl_path)
        self.assertTrue(os.path.isfile(custom_tpl_path))

        gen_loaded = TemplatePromptGenerator(template_file_path=custom_tpl_path)
        self.assertIn("Uji coba template X berhasil.", gen_loaded.id_templates)

    def test_export_three_splits(self):
        # 1. Setup 6 images: 3 train, 2 val, 1 test
        train_names = ["img_tr1.jpg", "img_tr2.jpg", "img_tr3.jpg"]
        val_names = ["img_va1.jpg", "img_va2.jpg"]
        test_names = ["img_te1.jpg"]

        all_names = train_names + val_names + test_names
        for name in all_names:
            with open(os.path.join(self.dm.frames_dir, name), "wb") as f:
                f.write(b"data")
            anno = self.dm.get_annotation(name)
            anno["boxes"] = [{"label": "pallet", "box_2d": [100, 100, 200, 200]}]
            anno["prompts"] = [
                {
                    "id": f"p_pos_{name}",
                    "user": f"Temukan pallet di {name}",
                    "assistant": json.dumps({"target_detected": True, "label": "pallet", "bounding_box": [100, 100, 200, 200], "frontier_score": None})
                },
                {
                    "id": f"p_neg_{name}",
                    "user": f"Temukan kursi di {name}",
                    "assistant": json.dumps({"target_detected": False, "label": None, "bounding_box": None, "frontier_score": 0.9})
                }
            ]
            self.dm.save_annotation(name, anno)

        # Set 3-way split explicitly
        self.dm.set_splits({
            "train": train_names,
            "val": val_names,
            "test": test_names
        })

        # 2. Export
        out_dir = os.path.join(self.test_dir, "export_3splits")
        ok, msg, stats = DatasetExporter.export(
            dataset_manager=self.dm,
            output_dir=out_dir,
            format_type="qwen",
            copy_images=False,
        )

        self.assertTrue(ok)
        self.assertEqual(stats["train_images"], 3)
        self.assertEqual(stats["val_images"], 2)
        self.assertEqual(stats["test_images"], 1)
        self.assertEqual(stats["train_samples"], 6)
        self.assertEqual(stats["val_samples"], 4)
        self.assertEqual(stats["test_samples"], 2)

        # 3. Verify output files
        train_file = os.path.join(out_dir, "train.jsonl")
        val_file = os.path.join(out_dir, "val.jsonl")
        test_file = os.path.join(out_dir, "test.jsonl")

        self.assertTrue(os.path.isfile(train_file))
        self.assertTrue(os.path.isfile(val_file))
        self.assertTrue(os.path.isfile(test_file))

        with open(train_file, "r") as f:
            tr_lines = [json.loads(l) for l in f if l.strip()]
        with open(val_file, "r") as f:
            va_lines = [json.loads(l) for l in f if l.strip()]
        with open(test_file, "r") as f:
            te_lines = [json.loads(l) for l in f if l.strip()]

        self.assertEqual(len(tr_lines), 6)
        self.assertEqual(len(va_lines), 4)
        self.assertEqual(len(te_lines), 2)

        # 4. Verify ZERO data leakage across all 3 sets
        def get_images_from_entries(entries):
            imgs = set()
            for e in entries:
                imgs.add(os.path.basename(e["messages"][0]["content"][0]["image"]))
            return imgs

        tr_imgs = get_images_from_entries(tr_lines)
        va_imgs = get_images_from_entries(va_lines)
        te_imgs = get_images_from_entries(te_lines)

        self.assertEqual(len(tr_imgs.intersection(va_imgs)), 0)
        self.assertEqual(len(tr_imgs.intersection(te_imgs)), 0)
        self.assertEqual(len(va_imgs.intersection(te_imgs)), 0)
        self.assertEqual(tr_imgs, set(train_names))
        self.assertEqual(va_imgs, set(val_names))
        self.assertEqual(te_imgs, set(test_names))

    def test_video_role_based_auto_split(self):
        """Test that 'Train & Val' videos split only into train/val, while 'Test Only' videos go 100% into test."""
        from PIL import Image
        dm = DatasetManager(self.test_dir)

        luar_frames = []
        for i in range(1, 11):
            fname = f"amr_frame_video_luar_{i:05d}.jpg"
            Image.new("RGB", (100, 100), color="green").save(os.path.join(dm.frames_dir, fname))
            if i <= 6:
                dm.save_annotation(fname, {
                    "image_file": fname,
                    "width": 100,
                    "height": 100,
                    "boxes": [{"id": f"b_{i}", "label": "pallet", "box_2d": [100, 100, 300, 300]}],
                    "prompts": []
                })
            luar_frames.append(fname)

        lab_frames = []
        for i in range(1, 6):
            fname = f"amr_frame_video_lab_{i:05d}.jpg"
            Image.new("RGB", (100, 100), color="blue").save(os.path.join(dm.frames_dir, fname))
            dm.save_annotation(fname, {
                "image_file": fname,
                "width": 100,
                "height": 100,
                "boxes": [{"id": f"lab_{i}", "label": "human", "box_2d": [200, 200, 400, 400]}],
                "prompts": []
            })
            lab_frames.append(fname)

        dm.register_video_frames("video_luar.mp4", "train_val", luar_frames)
        dm.register_video_frames("video_lab.mp4", "test", lab_frames)

        self.assertTrue(dm.has_test_only_frames())
        counts = dm.get_role_counts()
        self.assertEqual(counts["train_val"], 10)
        self.assertEqual(counts["test"], 5)

        # Run auto-split (e.g. 80% Train, 20% Val, 0% Test spinbox)
        splits = dm.auto_split(train_ratio=0.80, val_ratio=0.20, test_ratio=0.0, stratify=True, seed=42)

        # Verify 100% of lab frames are in test and 0% in train/val
        self.assertEqual(set(splits["test"]), set(lab_frames))
        self.assertEqual(len(set(splits["train"]).intersection(set(lab_frames))), 0)
        self.assertEqual(len(set(splits["val"]).intersection(set(lab_frames))), 0)

        # Verify 100% of luar frames are partitioned between train and val (8 train, 2 val)
        self.assertEqual(set(splits["train"]) | set(splits["val"]), set(luar_frames))
        self.assertEqual(len(splits["train"]), 8)
        self.assertEqual(len(splits["val"]), 2)

        # Verify persistence across reload
        dm2 = DatasetManager(self.test_dir)
        self.assertEqual(dm2.get_frame_role(lab_frames[0]), "test")
        self.assertEqual(dm2.get_frame_role(luar_frames[0]), "train_val")


if __name__ == "__main__":
    unittest.main()
