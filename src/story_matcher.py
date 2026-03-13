import nltk
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Paths
BASE_PATH       = "C:\\Users\\vijay\\OneDrive\\Desktop\\anime-project"
COMBINED_CSV    = BASE_PATH + "\\src\\combined_data.csv"
EMBEDDINGS_PATH = BASE_PATH + "\\src\\combined_embeddings.npy"

# Load data
print("🔵 Loading data for context aware matching...")
merged_df          = pd.read_csv(COMBINED_CSV)
caption_embeddings = np.load(EMBEDDINGS_PATH)
model              = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Data loaded!")

# ── Step 1: Smart Sentence Splitter ─────────────────────────────
def split_story_smart(story_text):
    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

    # Split into sentences
    sentences = nltk.sent_tokenize(story_text)

    # Clean sentences
    cleaned = []
    for s in sentences:
        s = s.strip()
        if len(s) > 3:
            cleaned.append(s)

    print(f"\n📌 Story split into {len(cleaned)} scenes:")
    for i, s in enumerate(cleaned):
        print(f"   Scene {i+1}: {s}")

    return cleaned

# ── Step 2: Context Aware Matching ──────────────────────────────
def find_matching_video_context(sentence, story_context,
                                 scene_index, total_scenes,
                                 used_videos):
    # Build context enriched query
    # Combine current sentence with story context
    # for better understanding
    context_query = f"{story_context} {sentence}"

    # Encode context enriched query
    query_embedding = model.encode([context_query])

    # Get similarities
    similarities = util.cos_sim(
        query_embedding, caption_embeddings
    )[0].numpy()

    # Penalize already used videos to avoid repetition
    for used_idx in used_videos:
        similarities[used_idx] *= 0.1

    # Get top 5 matches
    top_indices = np.argsort(similarities)[::-1][:5]

    # Pick best unused match
    best_idx   = top_indices[0]
    best_match = merged_df.iloc[best_idx]

    return {
        'video_id'  : best_match['video_id'],
        'caption'   : best_match['caption'],
        'source'    : best_match.get('source', 'msrvtt'),
        'similarity': float(similarities[best_idx]),
        'index'     : int(best_idx)
    }

# ── Step 3: Full Story to Video Sequence ────────────────────────
def match_story_to_videos(story_text):
    sentences    = split_story_smart(story_text)
    used_videos  = []
    matched      = []

    print(f"\n🔵 Context aware matching started...")

    for i, sentence in enumerate(sentences):
        # Build running story context
        # Use previous 2 sentences as context
        if i == 0:
            context = ""
        elif i == 1:
            context = sentences[0]
        else:
            context = " ".join(sentences[max(0, i-2):i])

        print(f"\n   📌 Scene {i+1}/{len(sentences)}")
        print(f"   Sentence : {sentence}")
        print(f"   Context  : {context[:50]}..." if len(context) > 50 else f"   Context  : {context}")

        match = find_matching_video_context(
            sentence, context,
            i, len(sentences),
            used_videos
        )

        # Track used video index to avoid repetition
        used_videos.append(match['index'])

        print(f"   Matched  : {match['video_id']}")
        print(f"   Caption  : {match['caption']}")
        print(f"   Source   : {match['source']}")
        print(f"   Score    : {match['similarity']:.4f}")

        matched.append({
            'scene'    : i + 1,
            'sentence' : sentence,
            'video_id' : match['video_id'],
            'caption'  : match['caption'],
            'source'   : match['source'],
            'score'    : match['similarity']
        })

    print(f"\n✅ Matched {len(matched)} scenes successfully!")
    return matched

# ── Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    story = (
        "A hero picks up his sword. "
        "He runs through the forest. "
        "He fights the enemy. "
        "He wins the battle. "
        "He celebrates his victory."
    )

    print("=" * 60)
    print("🎬 CONTEXT AWARE STORY MATCHER")
    print("=" * 60)

    results = match_story_to_videos(story)

    print("\n" + "=" * 60)
    print("📋 FINAL SCENE SEQUENCE:")
    print("=" * 60)
    for r in results:
        print(f"\n   Scene {r['scene']}: {r['sentence']}")
        print(f"   Video   : {r['video_id']} ({r['source']})")
        print(f"   Caption : {r['caption']}")
        print(f"   Score   : {r['score']:.4f}")

    print("\n✅ Context aware matching completed!")