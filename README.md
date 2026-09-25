# 🤖 AMR VLM Dataset Maker (Desktop App)

Aplikasi Desktop Python Modern untuk membuat dan mengurasi dataset **Vision-Language Model (VLM)** khusus **Object Grounding AMR (Autonomous Mobile Robot)** dengan bantuan Gemini AI.

---

## ✨ Fitur Utama

1. **🎬 Menu 1: Video to Frames Extractor**
   - Ekstraksi frame video kamera robot secara presisi berdasarkan **Interval Frame (tiap N frame)** atau **Interval Waktu (tiap X ms)**.
   - Opsi pembatasan resolusi (*smart resize*) agar ukuran file optimal untuk training VLM.
   - Multithreaded background processing dengan progress bar real-time.

2. **🏷️ Menu 2: Frame Curation & Gemini AI Auto-Grounding**
   - **Canvas Interaktif**: Zoom, Pan, gambar bounding box manual, drag & resize box dengan corner handles.
   - **Gemini AI Auto-Detect**: 1-klik untuk mendeteksi objek AMR (seperti *pallet, forklift, worker, charging dock, obstacle*) beserta koordinat spasial normalisasi 0–1000.
   - **Kurasi Cepat**: Hapus frame blur atau frame yang tidak berguna secara instan dengan tombol `Delete`.
   - **Keyboard Shortcuts**: Navigasi cepat dengan tombol `A` (Previous frame), `D` (Next frame), `Del` (Hapus).

3. **✍️ Menu 3: Prompt Editor, Augmentasi & Ekspor Dataset**
   - **Input Manual & Custom**: Tulis prompt pertanyaan user secara bebas per frame / per objek.
   - **Prompt Duplication**: Gandakan variasi pertanyaan untuk 1 gambar dengan koordinat objek yang sama.
   - **Negative Sample ("null")**: Generator sampel negatif untuk objek yang tidak ada di frame dengan output model `"null"`.
   - **Ekspor Siap Latih**:
     - Format **Qwen2-VL** (`messages: [{"role": "user"}, {"role": "assistant"}]`).
     - Format **ShareGPT / LLaVA** (`conversations: [{"from": "human"}, {"from": "gpt"}]`).
     - Train / Validation Split slider otomatis (misal: 80% Train, 20% Val).

4. **🌓 Modern UI dengan Light / Dark Mode Toggle**
   - Tampilan antarmuka modern (Dark & Light theme) yang bisa diganti kapan saja secara instan.

---

## 🚀 Cara Instalasi & Menjalankan

### 1. Buat & Aktifkan Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependensi
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi
```bash
python3 main.py
```
*(Atau tentukan folder project langsung)*:
```bash
python3 main.py --project ./amr_dataset_project
```

---

## ⚙️ Pengaturan Gemini API Key
1. Buka [Google AI Studio](https://aistudio.google.com/) dan buat API Key gratis.
2. Di aplikasi, klik tombol **⚙️ Gemini API** di pojok kanan atas.
3. Masukkan API Key Anda lalu klik **Simpan**.

---

## 📁 Struktur Dataset Hasil Export

```text
exports/
├── train.jsonl             # Data training
├── val.jsonl               # Data validation
├── dataset_info.json       # Ringkasan statistik dataset
└── images/                 # Gambar-gambar frame dataset
    ├── amr_frame_00001.jpg
    └── ...
```

### Contoh Format Sampel JSONL:
```json
{"id": "amr_frame_00001_v01", "image": "images/amr_frame_00001.jpg", "messages": [{"role": "user", "content": "<image>\nDeteksi pallet."}, {"role": "assistant", "content": "[620, 310, 850, 540]"}]}
{"id": "amr_frame_00001_v02", "image": "images/amr_frame_00001.jpg", "messages": [{"role": "user", "content": "<image>\nCari lokasi charging dock."}, {"role": "assistant", "content": "null"}]}
```
