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
        self.assertTrue(any('"null"' in l for l in lines))
        self.assertTrue(any('[620, 310, 850, 540]' in l for l in lines))

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
                train_imgs.add(data["image"])

        val_imgs = set()
        with open(os.path.join(out_dir, "val.jsonl"), "r") as f:
            for line in f:
                data = json.loads(line)
                val_imgs.add(data["image"])

        # Strictly verify no intersection between train and val images (NO LEAKAGE)
        intersection = train_imgs.intersection(val_imgs)
        self.assertEqual(len(intersection), 0, f"Data leakage detected! Images in both train and val: {intersection}")
        self.assertEqual(len(train_imgs) + len(val_imgs), 4)

if __name__ == "__main__":
    unittest.main()
