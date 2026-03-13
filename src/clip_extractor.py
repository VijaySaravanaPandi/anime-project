import subprocess
import os
import json
import pandas as pd
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer, util

# Paths
MERGED_CSV      = "C:\\Users\\vijay\\OneDrive\\Desktop\\anime-project\\src\\merged_data.csv"
EMBEDDINGS_PATH = "C:\\Users\\vijay\\OneDrive\\Desktop\\anime-project\\src\\caption_embeddings.npy"
VIDEOS_PATH     = "C:\\Users\\vijay\\OneDrive\\Desktop\\anime-project\\datasets-project\\data\\MSRVTT\\MSRVTT\\videos\\all"
TEMP_PATH       = "C:\\Users\\vijay\\OneDrive\\Desktop\\anime-project\\temp"
ANNOTATION_FILE = "C:\\Users\\vijay\\OneDrive\\Desktop\\anime-project\\datasets-project\\data\\MSRVTT\\MSRVTT\\annotation\\MSR_VTT.json"
FFMPEG_PATH     = "C:\\ffmpeg\\ffmpeg-8.0.1-essentials_build\\bin\\ffmpeg.exe"

# Load data
print("Loading data...")
merged_df          = pd.read_csv(MERGED_CSV)
caption_embeddings = np.load(EMBEDDINGS_PATH)
model              = SentenceTransformer('all-MiniLM-L6-v2')

# Load video timestamps from JSON
print("Loading video timestamps...")
with open(ANNOTATION_FILE, "r") as f:
    raw_data = json.load(f)

# Build timestamp lookup
timestamp_lookup = {}
for video in raw_data['images']:
    vid_id = video['id']
    timestamp_lookup[vid_id] = {
        'start': video.get('start time', 0),
        'end'  : video.get('end time', 10)
    }

# Function to find matching video
def find_matching_video(sentence):
    query_embedding = model.encode([sentence])
    similarities    = util.cos_sim(query_embedding, caption_embeddings)[0]
    best_idx        = int(np.argmax(similarities))
    matched         = merged_df.iloc[best_idx]
    return {
        "video_id"  : matched['video_id'],
        "caption"   : matched['caption'],
        "similarity": float(similarities[best_idx])
    }

# Function to extract video clip
def extract_clip(video_id, clip_index, duration_per_scene):
    input_path  = os.path.join(VIDEOS_PATH, f"{video_id}.mp4")
    output_path = os.path.join(TEMP_PATH, f"clip_{clip_index}.mp4")

    # Check if video exists
    if not os.path.exists(input_path):
        print(f"   ❌ Video not found: {input_path}")
        return None

    # Get timestamps
    timestamps = timestamp_lookup.get(video_id, {'start': 0, 'end': 10})
    start      = timestamps['start']
    end        = timestamps['end']
    duration   = end - start

    if duration <= 0:
        duration = 10

    print(f"   📹 Extracting {video_id} | start:{start}s end:{end}s duration:{duration:.1f}s")

    # Extract and resize clip using full ffmpeg path
    command = [
        FFMPEG_PATH, "-y",
        "-i", input_path,
        "-ss", str(start),
        "-t", str(duration),
        "-vf", "scale=640:360",
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"   ✅ Clip saved  : clip_{clip_index}.mp4")
        return output_path
    else:
        print(f"   ❌ Error       : {result.stderr[-200:]}")
        return None

# Test with sample story
if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('punkt_tab')

    story    = "A boy walks into a forest. He finds a magical dragon. They become friends and fly together in the sky."
    duration = 30  # total duration in seconds

    sentences          = nltk.sent_tokenize(story)
    duration_per_scene = duration / len(sentences)

    print(f"\n🔵 Extracting clips for {len(sentences)} sentences...")
    extracted_clips = []

    for i, sentence in enumerate(sentences):
        print(f"\n📌 Scene {i+1}: {sentence}")
        match = find_matching_video(sentence)
        print(f"   Matched    : {match['video_id']}")
        print(f"   Caption    : {match['caption']}")
        print(f"   Similarity : {match['similarity']:.4f}")

        clip_path = extract_clip(match['video_id'], i, duration_per_scene)
        if clip_path:
            extracted_clips.append(clip_path)

    print(f"\n✅ Total clips extracted : {len(extracted_clips)}")
    print(f"✅ Clips saved in        : {TEMP_PATH}")
    print("✅ Clip extraction completed successfully!")