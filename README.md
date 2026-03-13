# 🎬 AnimaStudio: Dataset-Driven Generative AI Video Engine
### 🎓 6th Semester Generative AI Project

AnimaStudio is a high-performance generative video pipeline designed to transform unstructured text narratives into stylized cinematic videos. Unlike purely diffusion-based models that are computationally expensive for long-form video, this project implements a **Retrieval-Augmented Generation (RAG)** approach for video synthesis.

![AnimaStudio Banner](https://img.shields.io/badge/Status-6th--Semester--Project-e91e8c?style=for-the-badge)
![AI](https://img.shields.io/badge/Generative--AI-FAISS--%2B--CLIP-7c3aed?style=for-the-badge)

---

## 📖 Introduction
In the current landscape of Generative AI, creating consistent, long-duration video from text remains a challenge. AnimaStudio solves this by utilizing a massive curated dataset and a semantic search engine. It "hallucinates" a storyboard from your input and then "realizes" it by finding, grading, and stitching together high-relevance video segments.

---

## 📊 Dataset: MSR-VTT (Microsoft Research Video to Text)
This project utilizes the **MSR-VTT** dataset, a large-scale benchmark for video understanding.
- **Scale**: Contains 10,000 video clips.
- **Richness**: Each clip is annotated with 20 natural language captions (200,000 captions total).
- **Diversity**: Covers a wide variety of domains (sports, news, movies, cooking, etc.).
- **Why MSR-VTT?**: It provides the ground truth mapping between visual actions and textual descriptions, making it the perfect foundation for a retrieval-based generative model.

---

## 🔍 How It Works: The "Semantic Bridge"

### 1. The Role of Captions
Captions are the "connectors" between your story and the pixel data. 
- We pre-process all 200,000 captions using **Sentence-BERT (all-MiniLM-L6-v2)** to convert them into 384-dimensional dense vectors (embeddings).
- These embeddings represent the **semantic meaning** of the video rather than just keywords.

### 2. FAISS (Facebook AI Similarity Search)
To make the generation instantaneous, we use **FAISS**. 
- When you enter a story, the engine splits it into scene-level sentences.
- Each sentence is converted into a vector and "queried" against the FAISS index.
- FAISS performs an incredibly fast L2-distance search to find the clip whose caption most closely matches your story's intent.

### 3. Ollama (Query Expansion)
To bridge the gap between simple story text and detailed dataset captions, we use **Gemma 3 (via Ollama)**. The LLM expands your story sentence into a rich visual prompt (describing lighting, environment, and movement) before the search begins.

---

## 🎭 Visual Intelligence & Stylization
Once the best clips are retrieved, the engine applies:
- **Style Transformers**: OpenCV-based bilateral filters convert standard video into "Anime" or "Cartoon" styles.
- **Emotion Grading**: A sentiment analysis pass detects the "mood" of the scene (e.g., Joy, Tension, Urgency) and applies custom color look-up tables (LUTs) to reinforce the narrative.

---

## 🚀 Installation & Usage

1. **Clone & Install**:
   ```bash
   git clone https://github.com/VijaySaravanaPandi/anime-project.git
   cd anime-project
   pip install -r requirements.txt
   ```

2. **Run Server**:
   ```bash
   python src/app.py
   ```

---

## 🛠️ Technical Stack
- **NLP**: Sentence-Transformers (CLIP-Style), NLTK, Ollama
- **Vector DB**: FAISS
- **Computer Vision**: OpenCV (Artistic Filters), PIL
- **Video Processing**: MoviePy, FFmpeg

---
*Developed as a Semester project for Generative AI course.*
