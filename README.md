# 🎬 AnimaStudio: Dataset-Driven Generative AI Video Engine

> 6th Semester Generative AI Project — Built as part of my academic curriculum.

AnimaStudio is a high-performance generative video pipeline that transforms unstructured text narratives into stylized cinematic videos. Unlike computationally expensive diffusion models, this project implements a **Retrieval-Augmented Generation (RAG)** approach for video synthesis — finding and stylizing real video clips that semantically match your story.

---

## ⚡ Quick Demo

Write a story → Get a stylized video in seconds:

```
"A hero picks up a sword. He fights the villain bravely. The hero wins."
```

→ AnimaStudio splits into scenes → semantically searches 213,000+ video captions → extracts matching clips → applies anime/cartoon style → stitches into a final video.

---

## ⚠️ Prerequisites — Read Before Running

> Running the 3 commands below will **crash** without these steps first.

### 1. 🐍 Python 3.10+
Download from [python.org](https://www.python.org/downloads/)

### 2. 🎞 FFmpeg (Required for video encoding)
- Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- Extract and **add `ffmpeg/bin` to your System PATH**
- Verify with: `ffmpeg -version`

### 3. 📦 Python Packages
```bash
pip install -r requirements.txt
```

### 4. 📊 Pre-Built Dataset Indexes (Download Required)

The dataset files are **too large for GitHub** (~1 GB total). Download them from the link below and place them inside the `src/` folder:

> **📥 [Download Pre-Built Indexes from Google Drive](https://drive.google.com/drive/folders/1il5--Km9ujX9EP8ID4bINDTyjHaIEU4n?usp=drive_link)**

Files to place in `src/`:
| File | Size | Description |
|------|------|-------------|
| `combined_data.csv` | ~14 MB | Video captions from MSR-VTT + UCF-101 |
| `combined_embeddings.npy` | ~312 MB | Pre-computed sentence vectors |
| `combined_faiss.index` | ~312 MB | FAISS binary search index |
| `ucf101_captions.csv` | ~2.3 MB | UCF-101 action captions |

### 5. 📹 MSR-VTT Video Dataset

Download from [Microsoft Research](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/) and place at:
```
datasets-project/data/MSRVTT/MSRVTT/videos/all/
datasets-project/data/MSRVTT/MSRVTT/annotation/MSR_VTT.json
```

> **Note:** Without the video files, the app automatically falls back to a **comic panel renderer** (offline, no GPU) — you still get output!

### 6. 🤖 Ollama + Gemma3 (Optional — Improves Search Quality)
```bash
# Install Ollama from https://ollama.com/
ollama run gemma3:4b
```
Without Ollama, the app uses direct FAISS search (still works great).

---

## ✅ Verify Your Setup

Run the diagnostic checker to see exactly what's missing:
```bash
python check_setup.py
```

This will check all dependencies and tell you precisely what to fix.

---

## 🚀 Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/VijaySaravanaPandi/anime-project.git
cd anime-project

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run setup checker (tells you if anything is missing)
python check_setup.py

# 4. Start the server
python src/app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## 📖 How It Works

### The "Semantic Bridge" Architecture

```
Your Story Text
      │
      ▼
[NLTK] Split into scene-level sentences
      │
      ▼
[Ollama/Gemma3] Query expansion (optional)
      │
      ▼
[SentenceTransformer: all-MiniLM-L6-v2] → 384-dim vector
      │
      ▼
[FAISS IndexFlatIP] Cosine search → top-50 candidates
      │
      ▼
[Smart Picker] Keyword + Character role scoring → best match
      │
      ▼
[FFmpeg] Extract clip from MSR-VTT / UCF-101
      │
      ▼
[OpenCV] Style Transfer (anime / cartoon / sketch / vivid / pixar)
      │
      ▼
[OpenCV] Emotion Color Grading (joy / tension / urgency / build-up)
      │
      ▼
[FFmpeg + MoviePy] Stitch with fade transitions
      │
      ▼
🎬 Final animation_output.mp4 (1280×720 @ 30fps)
```

### Fallback: Comic Panel Generator
If dataset videos are unavailable, the app generates rich comic panels using **pure PIL+OpenCV** (no API, no GPU, works 100% offline):
- 7 dynamic environments (forest, castle, ocean, mountain, city, night, plains)
- 6 action-based character poses (fight, run, victory, pick, think, stand)
- Cinematic vignette + emotion color grading
- Caption bar with word wrapping

---

## 📊 Dataset: MSR-VTT

- **Scale**: 10,000 video clips with 20 natural language captions each (200,000 total)
- **Diversity**: Sports, news, movies, cooking, and more
- **Why MSR-VTT?** Provides ground-truth visual-textual mappings — perfect for RAG-based retrieval

---

## 📐 Architecture Diagrams

| Diagram | Description |
|---------|-------------|
| ![Architecture Diagram](diagrams/Architecture%20Diagram.png) | System architecture — LLM + FAISS + Video Engine |
| ![Architecture Design](diagrams/Architecture%20Design.png) | Sequential logic flow from story to animation |
| ![Experiment Design](diagrams/Experiment%20Design.png) | Testing and validation pipeline |

---

## 🎭 Visual Intelligence & Stylization

| Feature | Technology |
|---------|-----------|
| Style Transfer | OpenCV bilateral filters + edge masking |
| Emotion Grading | Sentiment-based BGR channel shifting |
| Query Expansion | Ollama + Gemma3 4B (local LLM) |
| Semantic Search | FAISS IndexFlatIP (~1–5ms per query) |

**5 Visual Styles:**
- 🎨 `cartoon` — Bold edges + vivid colors
- 🌸 `anime` — Smooth + high saturation
- 🖋 `sketch` — Pencil-drawn feel
- 🌈 `vivid` — Hyper-saturated cinematic
- ✨ `pixar` — Warm, detailed lighting

---

## 🛠️ Technical Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | Flask |
| NLP / Embedding | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS (`IndexFlatIP`) |
| LLM Query Expansion | Ollama + Gemma3 4B |
| Sentence Tokenization | NLTK punkt |
| Computer Vision | OpenCV |
| Image Generation | PIL/Pillow (comic fallback) |
| Video Processing | MoviePy + FFmpeg |
| Datasets | MSR-VTT + UCF-101 |
| Frontend | Vanilla HTML/CSS/JS |

---

## 📁 Project Structure

```
anime-project/
├── src/
│   ├── app.py                 # Core Flask app + full AI pipeline
│   ├── build_faiss_index.py   # One-time index builder
│   ├── ucf101_setup.py        # Dataset merger + embedding generator
│   ├── cartoon_converter.py   # Standalone cartoon filter
│   └── video_stitcher.py      # Standalone clip assembler
├── datasets-project/          # Video datasets (not on GitHub)
├── models/                    # Model weights (not on GitHub)
├── output/                    # Final video output
├── temp/                      # Intermediate workspace
├── diagrams/                  # Architecture diagrams
├── check_setup.py             # ✅ Setup diagnostic script
└── requirements.txt
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|---------|
| `FileNotFoundError: combined_data.csv` | Download pre-built indexes from the Google Drive link above |
| `FileNotFoundError: MSR_VTT.json` | Download MSR-VTT dataset and place in `datasets-project/data/` |
| FFmpeg not found | Add FFmpeg to System PATH and restart terminal |
| No clips generated | App will use comic panel fallback — still produces output |
| Slow generation | Normal on CPU — style transfer takes 2–5s per clip |

---

*Developed as a 6th Semester project for Generative AI course.*
