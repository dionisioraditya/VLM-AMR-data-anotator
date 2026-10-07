import os
import json
import base64
import shutil
import subprocess
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any, Tuple, Optional

class GeminiClient:
    """Multi-backend Client for Vision spatial object grounding (Antigravity CLI or Google AI Studio)."""

    BACKEND_AGY = "agy"
    BACKEND_API = "api"

    AGY_MODELS = [
        "gemini-3.8-flash-low",
        "gemini-3.8-flash-medium",
        "gemini-3.8-flash-high",
        "gemini-3.7-flash-medium",
        "gemini-3.6-flash-low",
        "gemini-3.6-flash-medium",
        "gemini-3.6-flash-high",
        "gemini-3.1-pro-low",
        "gemini-3.1-pro-high",
        "claude-sonnet-5-5-low",
        "claude-sonnet-5-5-medium",
        "claude-sonnet-5-5-high",
        "claude-opus-5-5-low",
        "claude-opus-5-5-medium",
        "claude-opus-5-5-high",
        "gpt-oss-120b-medium",
    ]

    API_MODELS = [
        "gemini-3.6-flash",
        "gemini-flash-lite-latest",
        "gemini-3.8-flash",
    ]

    DEFAULT_MODELS = AGY_MODELS

    @staticmethod
    def find_agy_path() -> Optional[str]:
        """Locates the 'agy' binary in system path or standard user install locations."""
        p = shutil.which("agy")
        if p and os.path.isfile(p):
            return p
        user_p = os.path.expanduser("~/.local/bin/agy")
        if os.path.isfile(user_p):
            return user_p
        system_p = "/usr/local/bin/agy"
        if os.path.isfile(system_p):
            return system_p
        return None

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, backend: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        # Prefer Antigravity CLI if available on machine
        if backend:
            self.backend = backend
        else:
            self.backend = self.BACKEND_AGY if self.find_agy_path() else self.BACKEND_API

        if model:
            self.model = model
        else:
            self.model = self.AGY_MODELS[0] if self.backend == self.BACKEND_AGY else self.API_MODELS[0]

    def set_backend(self, backend: str):
        self.backend = backend
        if backend == self.BACKEND_AGY:
            if self.model not in self.AGY_MODELS:
                self.model = self.AGY_MODELS[0]
        else:
            if self.model not in self.API_MODELS:
                self.model = self.API_MODELS[0]

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip()

    def set_model(self, model: str):
        self.model = model.strip()

    def detect_objects(
        self,
        image_path: str,
        target_classes: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None,
        max_retries: int = 3,
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Detect objects with 2D bounding boxes using either Antigravity CLI or Gemini REST API.

        Returns:
            Tuple: (success: bool, message: str, boxes: List[Dict])
        """
        if self.backend == self.BACKEND_AGY:
            return self._detect_via_agy(image_path, target_classes, custom_instructions)
        else:
            return self._detect_via_api(image_path, target_classes, custom_instructions, max_retries)

    def _detect_via_agy(
        self,
        image_path: str,
        target_classes: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None,
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """Detect objects using local Antigravity CLI (agy)."""
        agy_bin = self.find_agy_path()
        if not agy_bin:
            return False, "Binary Antigravity CLI ('agy') tidak ditemukan di sistem.", []

        if not os.path.isfile(image_path):
            return False, f"Gambar tidak ditemukan: {image_path}", []

        class_hint = ""
        if target_classes and len(target_classes) > 0:
            class_hint = f"Focus strictly on detecting and finding bounding boxes for these classes: {', '.join(target_classes)}."

        extra_inst = custom_instructions or ""

        prompt_text = (
            f"You are an expert Autonomous Mobile Robot (AMR) vision grounding assistant.\n"
            f"Inspect the image file at '{os.path.abspath(image_path)}'.\n"
            f"{class_hint}\n"
            f"{extra_inst}\n"
            f"For every detected object, provide standard lowercase class 'label' and 'box_2d' as [ymin, xmin, ymax, xmax] "
            f"normalized to integer scale 0 to 1000, where 0 is top/left and 1000 is bottom/right.\n"
            f"Output strictly a raw JSON array of objects. Do not include markdown codeblocks (no ```json), explanations, or conversational filler.\n"
            f"Example format:\n"
            f"[{{\"label\": \"trash_bin\", \"box_2d\": [190, 241, 915, 508]}}]\n"
            f"If no relevant objects are detected, return an empty array: []"
        )

        cmd = [
            agy_bin,
            "-p", prompt_text,
            "--dangerously-skip-permissions",
            "--output-format", "text",
            "--disable-slash-commands",
        ]
        if self.model and self.model in self.AGY_MODELS:
            cmd.extend(["--model", self.model])

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if res.returncode != 0:
                err_text = res.stderr.strip() or res.stdout.strip()
                return False, f"agy CLI error (code {res.returncode}): {err_text[:150]}", []

            output = res.stdout.strip()
            # Clean markdown code blocks if agent wrapped it
            if output.startswith("```json"):
                output = output[7:]
            elif output.startswith("```"):
                output = output[3:]
            if output.endswith("```"):
                output = output[:-3]
            output = output.strip()

            # Find JSON array bracket if extra text exists
            start_bracket = output.find("[")
            end_bracket = output.rfind("]")
            if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
                output = output[start_bracket:end_bracket+1]

            parsed = json.loads(output)
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

            return True, f"Berhasil mendeteksi {len(valid_boxes)} objek via agy CLI.", valid_boxes

        except subprocess.TimeoutExpired:
            return False, "Waktu tunggu agy CLI habis (timeout 120s).", []
        except json.JSONDecodeError as e:
            return False, f"Gagal membaca format JSON dari respons agy: {str(e)}", []
        except Exception as e:
            return False, f"Gagal menjalankan agy CLI: {str(e)}", []

    def _detect_via_api(
        self,
        image_path: str,
        target_classes: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None,
        max_retries: int = 3,
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:

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
        req_data = json.dumps(payload).encode("utf-8")

        for attempt in range(max_retries):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            try:
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
                # 404: model deprecated or sunset -> switch immediately to gemini-3.6-flash
                if e.code == 404 and self.model != "gemini-3.6-flash":
                    self.model = "gemini-3.6-flash"
                    continue

                # 503: high demand spike -> wait and retry, or fallback if last attempt
                if e.code == 503:
                    if attempt < max_retries - 1:
                        time.sleep(2.5 * (attempt + 1))
                        continue
                    elif self.model != "gemini-flash-lite-latest":
                        # Final fallback to lighter model
                        self.model = "gemini-flash-lite-latest"
                        time.sleep(1.5)
                        continue

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

        return False, "Gagal setelah beberapa kali mencoba karena server Google sedang sibuk (503).", []
