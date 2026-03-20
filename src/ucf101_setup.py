import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths — resolved dynamically so this works on any machine
_HERE       = os.path.dirname(os.path.abspath(__file__))   # src/
_BASE       = os.path.dirname(_HERE)                        # project root
UCF101_PATH = os.path.join(_BASE, "datasets-project", "UCF101", "UCF-101")

print("🔵 Scanning UCF101 dataset...")

videos = []
categories = os.listdir(UCF101_PATH)
print(f"✅ Total categories found: {len(categories)}")

for category in categories:
    category_path = os.path.join(UCF101_PATH, category)
    if not os.path.isdir(category_path):
        continue

    # Convert CamelCase folder name to readable caption
    # e.g. "HorseRiding" → "a person doing horse riding"
    caption = "a person doing " + " ".join(
        ''.join([' ' + c if c.isupper() else c
                 for c in category]).strip().lower().split()
    )

    for video_file in os.listdir(category_path):
        if video_file.endswith(".avi") or video_file.endswith(".mp4"):
            full_path = os.path.join(category_path, video_file)
            videos.append({
                'video_id' : video_file.replace(".avi","").replace(".mp4",""),
                'caption'  : caption,
                'category' : category,
                'path'     : full_path
            })

# Save UCF101 captions
ucf_df = pd.DataFrame(videos)
ucf_df.to_csv(
    os.path.join(_HERE, "ucf101_captions.csv"),
    index=False
)

print(f"✅ Total UCF101 videos    : {len(ucf_df)}")
print(f"✅ Total categories       : {ucf_df['category'].nunique()}")
print(f"\n📌 Sample captions:")
print(ucf_df[['video_id','caption','category']].head(10).to_string())

# ── Merge with MSR-VTT ──────────────────────────────────────────
print("\n🔵 Merging MSR-VTT + UCF101...")

msrvtt_df = pd.read_csv(
    os.path.join(_HERE, "merged_data.csv")
)

# Keep only needed columns
msrvtt_df = msrvtt_df[['video_id','caption']].copy()
ucf_small = ucf_df[['video_id','caption']].copy()

# Add source tag
msrvtt_df['source'] = 'msrvtt'
ucf_small['source'] = 'ucf101'

# Combine
combined_df = pd.concat([msrvtt_df, ucf_small], ignore_index=True)
combined_df.to_csv(
    os.path.join(_HERE, "combined_data.csv"),
    index=False
)

print(f"✅ MSR-VTT videos         : {len(msrvtt_df)}")
print(f"✅ UCF101 videos          : {len(ucf_small)}")
print(f"✅ Combined total         : {len(combined_df)}")

# ── Generate Combined Embeddings ────────────────────────────────
print("\n🔵 Generating combined embeddings (this may take 10-15 mins)...")
model      = SentenceTransformer('all-MiniLM-L6-v2')
captions   = combined_df['caption'].tolist()
embeddings = model.encode(captions, show_progress_bar=True,
                          batch_size=64)
np.save(
    os.path.join(_HERE, "combined_embeddings.npy"),
    embeddings
)

print("\n✅ Combined embeddings saved!")
print("✅ UCF101 setup completed successfully!")
print("\n📌 Action categories available:")
for cat in sorted(categories)[:20]:
    print(f"   → {cat}")
print(f"   ... and {len(categories)-20} more")