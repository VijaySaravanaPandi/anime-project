import json
import pandas as pd
import numpy as np
import nltk
import os
from sentence_transformers import SentenceTransformer, util

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Paths
MERGED_CSV = "C:\\Users\\vijay\\OneDrive\\Desktop\\anime-project\\src\\merged_data.csv"
EMBEDDINGS_PATH = "C:\\Users\\vijay\\OneDrive\\Desktop\\anime-project\\src\\caption_embeddings.npy"

# Load merged data
print("Loading merged dataset...")
merged_df = pd.read_csv(MERGED_CSV)

# Load sentence transformer model
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate or load caption embeddings
if os.path.exists(EMBEDDINGS_PATH):
    print("Loading existing embeddings...")
    caption_embeddings = np.load(EMBEDDINGS_PATH)
else:
    print("Generating caption embeddings (this may take a few minutes)...")
    captions = merged_df['caption'].tolist()
    caption_embeddings = model.encode(captions, show_progress_bar=True)
    np.save(EMBEDDINGS_PATH, caption_embeddings)
    print("✅ Embeddings saved!")

# Function to split story into sentences
def split_story(story_text):
    sentences = nltk.sent_tokenize(story_text)
    print(f"\n📌 Story split into {len(sentences)} sentences:")
    for i, s in enumerate(sentences):
        print(f"   {i+1}. {s}")
    return sentences

# Function to find best matching video for a sentence
def find_matching_video(sentence):
    query_embedding = model.encode([sentence])
    similarities = util.cos_sim(query_embedding, caption_embeddings)[0]
    best_idx = int(np.argmax(similarities))
    matched = merged_df.iloc[best_idx]
    return {
        "video_id"  : matched['video_id'],
        "caption"   : matched['caption'],
        "similarity": float(similarities[best_idx])
    }

# Test with sample story
if __name__ == "__main__":
    story = "A boy walks into a forest. He finds a magical dragon. They become friends and fly together in the sky."

    print("\n🔵 Testing Story to Video Matching...")
    sentences = split_story(story)

    print("\n🔵 Finding matching videos...")
    for sentence in sentences:
        match = find_matching_video(sentence)
        print(f"\n✅ Sentence  : {sentence}")
        print(f"   Video ID  : {match['video_id']}")
        print(f"   Caption   : {match['caption']}")
        print(f"   Similarity: {match['similarity']:.4f}")

    print("\n✅ Video matching completed successfully!")