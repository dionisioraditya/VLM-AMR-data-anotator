import os
import json
import shutil
import tempfile
import unittest
from core.dataset_manager import DatasetManager
from core.exporter import DatasetExporter
from core.gemini_client import GeminiClient

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
                self.assertEqual(payload["bounding_box"], [620, 310, 850, 540])
                self.assertIsNone(payload["frontier_score"])
            else:
                has_negative = True
                self.assertIsNone(payload["bounding_box"])
                self.assertIsNotNone(payload["frontier_score"])
                self.assertAlmostEqual(payload["frontier_score"], 0.85)

        self.assertTrue(has_positive)
        self.assertTrue(has_negative)

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

if __name__ == "__main__":
    unittest.main()
