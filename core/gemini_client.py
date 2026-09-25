import os
import json
import base64
import urllib.request
import urllib.error
from typing import List, Dict, Any, Tuple, Optional

class GeminiClient:
    """Client for Gemini 2.0 Flash / 1.5 Flash spatial object grounding."""

    DEFAULT_MODELS = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-2.5-flash",
    ]

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip()

    def set_model(self, model: str):
        self.model = model.strip()

    def detect_objects(
        self,
        image_path: str,
        target_classes: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None,
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Detect objects with 2D bounding boxes using Gemini Vision.

        Returns:
            Tuple: (success: bool, message: str, boxes: List[Dict])
            Each box in boxes has {"label": str, "box_2d": [ymin, xmin, ymax, xmax]} (0-1000 scale)
        """
        if not self.api_key:
            return False, "API Key Gemini belum disetel. Silakan masukkan API Key di Pengaturan.", []

        if not os.path.isfile(image_path):
            return False, f"Gambar tidak ditemukan: {image_path}", []

        # Read and encode image to base64
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            return False, f"Gagal membaca file gambar: {str(e)}", []

        # Determine MIME type
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = "image/jpeg"
        if ext == ".png":
            mime_type = "image/png"
        elif ext == ".webp":
            mime_type = "image/webp"

        # Build prompt
        class_hint = ""
        if target_classes and len(target_classes) > 0:
            class_hint = f"Focus particularly on identifying these classes if present: {', '.join(target_classes)}."

        extra_inst = custom_instructions or ""

        prompt_text = f"""You are an expert Autonomous Mobile Robot (AMR) vision assistant.
Detect all key objects in the image such as pallets, wooden pallets, forklifts, human workers, charging docks, obstacles, cones, boxes, or AGVs/AMRs.
{class_hint}
{extra_inst}

For every detected object, provide:
1. "label": standard lowercase object class (e.g. "pallet", "forklift", "worker", "obstacle", "charging_dock")
2. "box_2d": [ymin, xmin, ymax, xmax] coordinates normalized to integer scale 0 to 1000, where 0 is top/left and 1000 is bottom/right.

Return strictly a JSON array of objects. Do not include any conversational filler or explanation.
Example output format:
[
  {{"label": "pallet", "box_2d": [620, 310, 850, 540]}},
  {{"label": "worker", "box_2d": [210, 750, 680, 890]}}
]
If no relevant objects are detected, return an empty array: []
"""

        # Prepare request payload for Gemini REST API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_image,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        }

        headers = {"Content-Type": "application/json"}

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_bytes = resp.read()
                resp_json = json.loads(resp_bytes.decode("utf-8"))

            candidates = resp_json.get("candidates", [])
            if not candidates:
                return False, "Gemini tidak mengembalikan kandidat respons.", []

            first_candidate = candidates[0]
            parts = first_candidate.get("content", {}).get("parts", [])
            if not parts:
                return False, "Respons Gemini kosong.", []

            raw_text = parts[0].get("text", "").strip()

            # Clean markdown codeblocks if present
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            parsed = json.loads(raw_text)
            if not isinstance(parsed, list):
                if isinstance(parsed, dict) and "objects" in parsed:
                    parsed = parsed["objects"]
                else:
                    parsed = [parsed]

            valid_boxes = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "object")).strip().lower()
                box_2d = item.get("box_2d")
                if isinstance(box_2d, list) and len(box_2d) == 4:
                    try:
                        box_ints = [max(0, min(1000, int(round(coord)))) for coord in box_2d]
                        valid_boxes.append({
                            "label": label,
                            "box_2d": box_ints,
                        })
                    except (ValueError, TypeError):
                        continue

            return True, f"Berhasil mendeteksi {len(valid_boxes)} objek.", valid_boxes

        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            try:
                err_json = json.loads(err_msg)
                detail = err_json.get("error", {}).get("message", err_msg)
            except Exception:
                detail = err_msg
            return False, f"HTTP Error {e.code}: {detail}", []
        except urllib.error.URLError as e:
            return False, f"Koneksi gagal: {str(e.reason)}", []
        except json.JSONDecodeError as e:
            return False, f"Gagal membaca format JSON respons: {str(e)}", []
        except Exception as e:
            return False, f"Terjadi kesalahan saat memanggil Gemini: {str(e)}", []
