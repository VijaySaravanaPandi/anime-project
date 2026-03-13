# 🎬 AnimaStudio: AI-Powered Story-to-Video Engine

AnimaStudio is a sophisticated generative video pipeline that transforms text-based stories into stylized cinematic animations. It combines semantic search (FAISS), natural language processing, and advanced computer vision techniques to match your narrative with real-world video datasets and apply artistic styles.

![AnimaStudio Banner](https://img.shields.io/badge/AnimaStudio-Creative--Engine-e91e8c?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Framework-000000?style=for-the-badge&logo=flask&logoColor=white)

---

## ✨ Features

- **📖 Intelligent Story Splitting**: Automatically breaks down long stories into logically consistent scenes.
- **🔍 Semantic Video Matching**: Uses **FAISS** and **Sentence Transformers** (CLIP-style) to find the most visually relevant clips for your story.
- **🤖 Ollama Integration**: Optionally enhances search queries using local LLMs (like Gemma 3) for better context awareness.
- **🎭 Artistic Stylization**: Apply "Anime", "Cartoon", "Vivid", or "Sketch" filters using OpenCV bilateral filters and color grading.
- **💓 Emotion-Driven Grading**: Dynamically adjusts video color palettes (e.g., warmer for victory, cooler for chase) based on the sentiment of the scene.
- **🛠️ Fully Local Pipeline**: Designed to run locally with FFmpeg and local datasets.

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10 - 3.13** (Recommended)
- **FFmpeg**: Must be installed and accessible.
- **Ollama** (Optional): For AI-enhanced query expansion.

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/anime-project.git
cd anime-project

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install flask pandas numpy faiss-cpu sentence-transformers opencv-python moviepy pillow nltk
```

### 3. Setup Paths
Open `src/app.py` and ensure the paths for `BASE_PATH`, `VIDEOS_PATH`, and `FFMPEG_PATH` match your local system:
```python
BASE_PATH   = "C:/path/to/your/project"
FFMPEG_PATH = "C:/path/to/ffmpeg.exe"
```

### 4. Run the Application
```bash
python src/app.py
```
Visit `http://localhost:5000` in your browser.

---

## 🏗️ Architecture

1. **Input**: User provides a story (e.g., "A knight enters the dark forest").
2. **Analysis**: The engine splits the story into scenes and uses Ollama to extract visual keywords.
3. **Retrieval**: FAISS searches the MSR-VTT dataset for the best matching video clips.
4. **Grading**: The system detects the "emotion" of the scene and applies a color grade.
5. **Assembly**: MoviePy stitches the clips together with smooth transitions and titles.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Search Engine**: FAISS (Facebook AI Similarity Search)
- **NLP**: Sentence-Transformers, NLTK
- **Computer Vision**: OpenCV, Pillow
- **Video Editing**: MoviePy, FFmpeg

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Created with ❤️ by AnimaStudio Team*
